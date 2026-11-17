# Parsing-cache contamination when one set of structure definitions parses many untrusted samples

## Summary

`dissect.cstruct` deliberately caches a lot on the *type* objects, because a definition is normally loaded once and
then reused for many samples. Three caches interact badly when a caller does what an analyst usually does — load a
definition once, then feed it a sequence of untrusted samples:

1. **The layout (field offsets and `__size__`) is derived from the types a structure embeds, and is only recomputed
   when that structure itself is committed.** A structure that embeds another type keeps the offsets and size computed
   against the *old* layout of that type.
2. **The compiled reader/writer bakes those offsets and sizes into generated source code**, and only regenerates when
   it is replaced — which happens on `commit()` of *that* structure, not of the embedded one.
3. **The lazily-built non-inlining fallback (`_Fallback`) memoises a generated function and never invalidates it**, so
   a second change to a nested type is silently ignored.

The result is that the *order in which samples were parsed* — and nothing else, since the bytes and the final
definition are identical — changes how a later sample is interpreted. The same bytes, replayed in a different order,
give different values. This is a genuine cross-sample state dependency, i.e. one untrusted sample influences the
interpretation of the next.

The contamination is **not** driven by the sample's payload bytes: nothing in the parser writes layout state from
decoded data. It is driven by *timing*: a sample that happens to be parsed before a definition update freezes the
caches that the update should have invalidated. Separating the two is the core of the diagnosis below.

## Reproduction environment

Everything below was run against `task/95` at `65eeecb` ("Add declarative syntax (#161)"):

```bash
python3 -m pytest tests -q     # 1071 passed, 45 skipped
```

All reproductions are self-contained `cstruct` scripts; none require fixtures or test helpers.

## Minimal difference

The smallest divergence needs a nested **union that cannot be inlined**, so that the outer structure reads it through
its own `_read` and stores the resulting layout in generated code:

```python
CDEF = """
union inner_u { uint8 a; };
union outer_u { inner_u i; uint8 b; };
struct test { uint8 x; outer_u ou; uint8 y; };
"""
BUF = bytes([0xAA, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99])
```

Grow `outer_u` by eight bytes, then parse the *same* `BUF`. Only the ordering of "parse a sample" vs "apply the
mutation" differs:

```python
def cell(compiled, mutate_first):
    cs = cstruct().load(CDEF, compiled=compiled)
    if mutate_first:
        cs.outer_u.add_field("wide", cs.uint64)
        return cs.test(BUF)
    cs.test(BUF)                              # a sample is parsed under the old layout
    cs.outer_u.add_field("wide", cs.uint64)
    return cs.test(BUF)
```

Measured result:

```
mode      order          (x,      y,      ou.i.a)     correct?
compiled  mutate-first   x=0xaa   y=0x22   ou.i.a=0x11   WRONG (stale offset)
compiled  parse-first    x=0xaa   y=0x99   ou.i.a=0x11   right (accidentally)
interp    mutate-first   x=0xaa   y=0x22   ou.i.a=0x11   WRONG (stale offset)
interp    parse-first    x=0xaa   y=0x22   ou.i.a=0x11   WRONG (stale offset)
```

The correct answer is computed by hand from the *final* definition:

```
union outer_u { inner_u i; uint8 b; uint64 wide; }   -> size 8, alignment 8
struct test { uint8 x; outer_u ou; uint8 y; }        -> x @ 0, ou @ 1..8, y @ 9, size 10
BUF = aa 11 22 33 44 55 66 77 88 99
      x=0xaa                        ou bytes = 11..88
                                y = BUF[9] = 0x99
```

So `y=0x99` is correct and `y=0x22` (the old field offset) is wrong. The order dependence is in the **compiled** row:
the same bytes and the same mutation give `0x22` or `0x99` depending only on whether a sample was parsed first.

The reason is worth stating precisely, because the "right" cell is right for the wrong reason:

* **compiled, mutate-first** — the method is generated *after* the mutation. `outer_u` is 8 bytes, but `test` was
  never re-derived, so the generator emits `stream.seek(start + 2)` from the stale pinned offset. `y` reads at 2.
* **compiled, parse-first** — the method was frozen *before* the mutation, and it reads `ou` through
  `_type1._read(stream, ...)` rather than from a fixed slice. That call now consumes 8 bytes, so the following
  `stream.read(1)` for `y` lands at 9 — the correct position, by accident.
* **interpreted** — both orders seek to the pinned offset 2, so both are identically wrong.

Neither path is a usable oracle: the compiled path is order-dependent, and the interpreted path is consistently stale.
The only reliable check is against the definition, which is why the harnesses below compute ground truth by hand.

### What the contaminated state looks like

Immediately after the mutation, the outer type is internally inconsistent — `outer_u` grew, but `test` did not:

```
before: test.__size__=3  ou.__size__=1  fields=[('x',0), ('ou',1), ('y',2)]
after : test.__size__=3  ou.__size__=8  fields=[('x',0), ('ou',1), ('y',2)]   <-- y still at 2
```

`test` now claims to be 3 bytes long while its own second field occupies bytes 1–8 and overlaps its third field. The
definition is no longer a valid layout, and `test.__size__` disagrees with the fields it contains. Calling
`cs.test.commit()` does **not** repair it, because `Field.offset` is treated as an *input* on re-commit (see
root cause 1).

## Root causes

### 1. `Field.offset` is both an input and an output, so re-commit cannot repair a layout

`Field` stores the resolved position in the same attribute the caller uses to request one
(`dissect/cstruct/types/structure.py:47`), and `_calculate_size_and_offsets` treats any non-`None` `offset` as a
declared offset that leads the layout (`structure.py:172`):

```python
for field in fields:
    if field.offset is not None:
        # If a field already has an offset, it's leading
        offset = field.offset
```

The loop then writes the derived offset back into the same field (`structure.py:219`). Once a structure has been laid
out, every field looks as though the caller had explicitly pinned it, so a second layout pass reproduces the first
pass's answer regardless of how the embedded types changed. Demonstrated directly:

```python
t = cs.test
cs.outer_u.add_field("wide", cs.uint64)   # outer_u is now 8 bytes
t.commit()                                # no effect
print([(f._name, f.offset) for f in t.__fields__])   # [('x',0), ('ou',1), ('y',2)]  still stale
for f in t.__fields__:
    f.offset = None                       # what a correct re-derive would have used
t.commit()
print([(f._name, f.offset) for f in t.__fields__])   # [('x',0), ('ou',1), ('y',9)]
```

Because the derived value is sticky, the layout is effectively write-once and the only thing that can change it is a
*brand-new* class — which is exactly what "mutate before the first parse" produces, and "mutate after" does not.

### 2. Committing a type does not re-derive the types that embed it

`commit()` (`structure.py:373`) recomputes `cls` and, if `cls` was compiled, recompiles `cls`. It never touches the
structures that embed `cls`. Since a structure's offsets and size are derived from the sizes of its field types
(`structure.py:153`), every ancestor now has a stale layout — and #1 guarantees that a later spontaneous re-commit
would not fix it.

The same gap applies to **array types**, which cache their `__size__` at creation
(`cstruct.py:524` comment: "Structures can be dynamically modified, so include the size and alignment in the cache
key"). An `inner[2]` created while `inner` was one byte keeps `__size__ == 2` forever. This produces a divergence that
needs no ordering at all, and **the two paths disagree in opposite directions**:

```python
CDEF = """
struct inner { uint8 a; };
struct test { uint8 x; inner arr[2]; uint8 y; };
"""
cs = cstruct().load(CDEF)
cs.inner.add_field("b", cs.uint8)     # one legitimate call, no samples parsed yet

# inner is now 2 bytes; a freshly built inner[2] is 4 bytes
cs.inner[2].__size__ == 4
cs.test.__fields__[1].type.__size__ == 2      # the cached array the field still holds
```

```
compiled=True   inner.size=2 arr.size=2 test.size=4  x=1 y=6
compiled=False  inner.size=2 arr.size=2 test.size=4  x=1 y=4
atomic reference (definition already has `b`):
compiled=True   x=1 y=6
compiled=False  x=1 y=6
```

Here the **compiled** reader is right (`y=6`) and the **interpreted** reader is wrong (`y=4`). The compiled reader
advances past the fields consecutively using their live sizes; the interpreted reader seeks to the pinned offset 3. So
"compiled vs interpreted" is not a safe oracle in either direction — both read the same corrupted layout state
differently.

### 3. The compiled method is frozen at first use, and its `_Fallback` never invalidates

`compile_structure` installs a `_LazyCompile` descriptor (`compiler.py:66`, `compiler.py:90`). The first attribute
access replaces the descriptor with the generated `classmethod`, baking the current layout into source text. This
happens on the first *read attempt*, and even a failed attempt freezes it:

```python
cs = cstruct().load(CDEF)
try:
    cs.test(b"")                       # EOFError: "Read 0 bytes, but expected 1"
except EOFError:
    pass
type(cs.test.__dict__["_read"]).__name__   # 'classmethod' -- already frozen
```

Merely reading `cs.test._read` does it too. After that, the only path back to a new layout is `commit()` on **that**
class — which, per #1 and #2, cannot repair it.

The `_Fallback` used when an inlined type changed (`compiler.py:185`) has the same defect one level deeper:

```python
def __call__(self, cls, stream, context, endian):
    if self.func is None:
        self.func = _ReadSourceGenerator(self.cls, self.fields, inline=False).generate().__func__
    return self.func(cls, stream, context=context, endian=endian)
```

It memoises both `self.fields` (a snapshot of `cls.__fields__` taken when the descriptor was built) and the generated
function. A **second** mutation therefore hits a cached function built for the first mutation:

```python
CDEF = """
struct inner { uint8 a; };
struct other { uint8 o; };
struct test { uint8 pre; inner in; uint8 post; other oth; };
"""
BUF = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])

cs = cstruct().load(CDEF)
cs.other.add_field("x", cs.uint8)   # trips the witness of `test` -> _Fallback generated here
cs.test(BUF)                        # pre=0x11 in.a=0x22 post=0x33  (oth.x=0x55)
cs.inner.add_field("b", cs.uint8)   # second mutation: fallback is already cached
cs.test(BUF)
```

```
compiled=True   post=0x44      (reads: pre=0x11 in.a=0x22 in.b=0x33 post=0x44 oth.o=0x55)
compiled=False  post=0x33      (reads: pre=0x11 in.a=0x22 in.b=0x33 post=0x33 oth.o=0x44)
```

With the final definition `post` is at offset 4, so `0x44` is correct and the compiled path is right here. The
divergence again shows that a sample parsed between two definition updates changes the result of a later sample.

### Interaction with context expressions

Dynamic array sizes are evaluated against a context built from the fields already read. In compiled code this is a
plain `dict` of locals (`_context`, `compiler.py`), and identifiers that are not locals go through
`_identifier(cs, context, ...)`. Crucially, `cs` here is captured **when the source was generated**:

```python
fn = cs.test.__dict__["_read"].__func__
fn.__globals__["cs"] is cs      # True: the generated function froze this instance
```

Because the context is assembled from the *readers'* view of previous fields, a stale layout changes what an
expression resolves to. The divergence below is a direct consequence of an outer type keeping the old offset of the
nested union, so the expression sees the old `n`:

```python
CDEF = """
union inner_u { uint8 n; };
union outer_u { inner_u i; uint8 b; };
struct test { uint8 x; outer_u ou; uint8 n; char d[n & 7]; uint8 y; };
"""
```

In the minimal case (`test { uint8 x; outer_u ou; uint8 y; }`) the same mechanism simply moves `y`; with an expression
in the structure it also silently changes the *length* of `d` and every field after it, so a contaminated parse can
hand back a shorter or longer buffer than the definition implies — with no error.

Note the expression path itself is not the bug: `Expression.source()` and `Expression.evaluate()` agree
(`tests/test_expression.py::test_expression_source_code`). The bug is that the context it is evaluated against is
produced by a stale layout.

## Distinguishing legitimate type updates from sample-driven pollution

These are separate axes and should not be conflated when triaging:

| | Legitimate update | Sample-driven contamination |
|---|---|---|
| Triggered by | `add_field`, `start_update`, `commit`, loading more definitions | Parsing a sample |
| Changes | `__fields__`, offsets, sizes, generated code | Member caches, lazily generated functions, `_Struct` LRU entries |
| Should be visible to later samples | Yes, immediately and atomically | Never |
| Observed in this audit | `test.__size__` stale after an embedded type grew (#1, #2) | compiled `_read` frozen by an earlier sample (#3) |

**No sample payload is written to layout state.** The parser never derives a field, offset or size from decoded bytes;
`_read` only produces a new object plus `__dynamic_sizes__` on *that* object. Every layout-affecting mutation in the
reproductions above comes from the `add_field` calls, which are the caller's deliberate type updates. So the failure
is not classic "poisoned sample rewrites the schema"; it is "the schema update is applied non-atomically, and a sample
parsed in the window between the sub-updates observes a half-applied schema." That distinction matters for the fix: it
cannot be solved by sanitising values, only by making the layout and its caches a coherent, versioned unit.

What *is* sample-driven, and should be separated from the layout bug, is benign but unbounded cache growth:

* **Enum/flag pseudo-members.** `Enum._missing_` and `Flag` construct and cache a member for every unknown value.
  Parsing 300 samples of a `flag F : uint64` grows `F._value2member_map_` from 2 to 257 entries. This is per-type and
  unbounded with attacker-chosen values for wide base types.
* **`_struct` LRU keyed on a sample-derived count.** A dynamic array of a packed type evaluates `n` and builds
  `struct.Struct(f"{count}{fmt}")`; `dissect/cstruct/util.py:57` caches those globally with `lru_cache(1024)`. Distinct
  sample counts evict each other (`currsize` grew by 43 for counts 1..39 in one structure) — legitimate caching, but
  churn is driven by sample content.
* **`cs.uint32[count]`** caches array types per instance (`cstruct.py:208`), bounded at 4096, same property.

None of these change a later sample's decoded values (verified: the `repr` of a fixed flag value is stable regardless
of parse order). They are resource-exhaustion concerns, not correctness ones, and are out of scope for the fix below.

## Replay methodology used

To separate "legitimate update" from "contamination" I compared cells that share bytes and final definition but differ
only in history. This is the property that must hold for a definition reused across untrusted samples:

> For a fixed final definition and a fixed byte string, the parse result must be independent of every sample parsed
> before it.

Four harnesses were used, all differential rather than assertion-based:

1. **Four-cell matrix** — compiled/interpreted × mutate-before-first-use/after-first-use.
2. **Order replay** — a single instance fed a permutation of samples, each result compared to a fresh instance parsing
   the same bytes (`order.py`). Baseline: 0 differences across the shapes tested; the divergence appears only when a
   *type mutation* is interleaved.
3. **Contamination fuzz** — random mutation sequences interleaved with random samples, compared to a fresh instance
   with the mutation set applied atomically. This is what produced the `outer_wide`/`test_extra` finding.
4. **Hand-computed ground truth** — for each minimal case, the layout is computed from the definition and the expected
   bytes indexed directly, so neither the compiled nor the interpreted path is trusted as an oracle.

The last point is important: as MRE-1 shows, either path can be the wrong one, so "compiled == interpreted" is
necessary but not sufficient.

## Minimal reproducer set

Three cases capture the distinct failure modes. All use a single `cstruct` instance and no test helpers.

**MRE-1 — array of a structure, one legitimate update, no ordering at all.** Offsets/size go stale; compiled and
interpreted then disagree with each other *and* one of them disagrees with the definition.

**MRE-2 — nested non-inlinable union, order dependent.** The minimal difference from the section above: same bytes,
same mutation, `y=0x22` vs `y=0x99`.

**MRE-3 — `_Fallback` memoisation.** Requires two mutations separated by a parse; the second is ignored by compiled
code while interpreted code sees it.

## Isolation design

The fix treats **layout + generated code + fallback** as one versioned unit, so a layout change is atomic from the
point of view of any reader. The reference implementation is in `union-cache-contamination.patch`, is ~140 lines, and
was validated against all three reproducers plus the harnesses above. The full test suite goes from 1071 passing to
1063 passing with 8 failures, all of which are tests asserting the *exact* generated source text (`test_lazy_compile`,
`test_compiled_union_source`, and the `test_generate_*_write` family) and therefore need the new guard line added to
their expectations; no behavioural test fails.

### 1. Split declared from derived offsets

`Field` gains `declared_offset` for the caller's request; `offset` remains the derived value. Layout reads
`declared_offset`; everything that reads positions reads `offset`. A re-commit then genuinely recomputes, which is the
precondition for every other part of the fix.

### 2. Record what the layout was derived from, and make it observable

`commit()` stamps each type with a monotonically increasing `__layout_generation__` and records, for every embedded
structure type (including through arrays), the generation it was laid out against. It also bumps a single
`cstruct._layout_version` counter, so the common "nothing changed" check is one attribute lookup and one integer
compare:

```python
def _layout_is_fresh(cls) -> bool:
    if cls.__layout_version__ == cls.__cs__._layout_version:
        return True                      # O(1) fast path
    return all(embedded.__layout_generation__ == generation
               for _, embedded, generation in cls.__layout_witness__)
```

### 3. Make a layout change propagate eagerly and atomically

`commit()` now also re-derives every structure in the instance whose witness is stale, to a fixed point. This is what
makes "mutate before first use" and "mutate after first use" indistinguishable: the layout of *every* type is already
current before any reader runs. Re-deriving also re-resolves cached array field types against the current element size,
because an array's `__size__` is fixed at creation and cannot otherwise be repaired.

### 4. Regenerate compiled code instead of freezing it

When a compiled method detects a layout change, it calls the same re-derive path and then regenerates itself, so the
generated offsets match the layout in force. Any *in-flight* call dispatches to the interpreted method for that one
read — the interpreted method reads the fresh layout dynamically, so it is correct for the current layout. This keeps
the two paths consistent at the transition instead of letting one run ahead:

```python
# generated preamble, both read and write
if cls._ensure_layout(): return <interpreted method>(cls, ...)
```

`_Fallback` is included: it records the generation of the layout its function was built for and rebuilds when that
generation changes, so a second mutation can no longer be ignored.

### Why both paths stay consistent

The property the design establishes is: *a reader either observes the layout it was generated for, or it observes the
current layout via a path that does not depend on generated offsets.* There is no third state. Concretely:

* A layout change is applied to the type object first, then to every embedder, then to cached array field types —
  all inside `commit()` before it returns.
* Compiled code cannot execute against a stale layout, because its first action is the freshness check, which either
  confirms the layout it was built for is still current or reroutes to the dynamic path.
* The interpreted reader has no baked offsets, so it is correct for whatever layout the type currently has; it needs
  no per-read check, which keeps its hot path free. Measured over three runs of a structure with an inlined structure,
  an inlined union, a char array and a dynamic array: compiled 8.6–8.8 µs/op before and 8.9–9.1 µs/op after
  (~3–4%, the added guard); interpreted 20.3–20.8 µs/op before and 20.2–21.5 µs/op after (noise).

### Validation of the design

Against the reference patch:

| Case | Before | After |
|---|---|---|
| MRE-2, compiled × (mutate-first, parse-first) | `0x22` / `0x99` (order dependent) | `0x99` / `0x99` |
| MRE-2, interpreted × (mutate-first, parse-first) | `0x22` / `0x22` (both stale) | `0x99` / `0x99` |
| MRE-1, compiled vs interpreted | `y=6` vs `y=4` | `y=6` vs `y=6`; `arr.__size__ == 4 == inner[2].__size__` |
| MRE-3, compiled vs interpreted | `post=0x44` vs `post=0x33` | both `post=0x44` |
| `test.__size__` right after the embedded type grows | `3` (stale, overlapping fields) | `10` |
| Order replay (`order.py`) | 0 differences (no interleaved mutation) | 0 |
| Contamination fuzz (`contam.py`, 400 trials) | 6 findings | 0 parse findings |
| Full test suite | 1071 pass | 1063 pass, 8 exact-source assertions to update |

The fuzz's only residual differences are re-serialising an object that was *constructed* before the layout changed;
that object legitimately predates the new layout and is a separate question from parsing a sample, so it is reported
rather than counted as contamination.

### Residual risks the isolation does not cover

* **Objects outliving a layout change.** An object parsed before a type update still holds the old `_buf`,
  `__dynamic_sizes__` and field set. Writing it after the update re-serialises it under the new layout and can raise
  or produce surprising bytes. The fix makes *parsing* order-independent; it does not version individual objects.
  Callers holding parsed objects across a definition change need to re-parse.
* **`_struct` / array / enum caches keyed on sample values** (see above): bounded but attacker-influenced; resource
  policy, not correctness.
* **`cstruct.copy()` is not a safe isolation boundary.** `copy.copy(cls)` on a class returns the class itself, so
  `cstruct.copy()` calls `copy.copy(type_)` which is a no-op, then rebinds the *original* types' `__cs__` to the copy
  (`cstruct.py:247`–`256`). After `cp = cs.copy()`, `cs.test is cp.test` and `cs.test.__cs__ is cp`, so mutating or
  reconfiguring the copy changes how the original parses (verified with a changed `#define` used in an array size:
  compiled and interpreted disagreed by exactly this mechanism: `d=b''` vs `d=b'AB'`). Using a copy per untrusted
  batch is therefore **not** an isolation mechanism today. If per-batch isolation is desired, constructing a fresh
  `cstruct` and loading the definition text is the only safe option.

## Recommendation

For callers processing untrusted samples today, and in addition to the fix:

1. Load the full definition **once**, before parsing anything, and do not mutate types while samples are in flight.
   If definitions must change, construct a new `cstruct` and re-load — do not rely on `add_field` to be atomic, and do
   not rely on `copy()` to isolate.
2. If a divergence is suspected, reproduce with the four-cell matrix in this document rather than comparing
   compiled to interpreted: either path can be stale, in opposite directions (MRE-1).
3. Treat enum/flag value-map growth and `_struct` LRU churn as separate resource concerns and cap them at the
   application level when samples are attacker-controlled.

## Reference patch

`union-cache-contamination.patch` (239 lines against `65eeecb`) is a validated reference implementation. It applies
cleanly to a pristine checkout and was verified end to end (`git apply`, then the reproducers and the suite). It
touches:

* `dissect/cstruct/types/structure.py` — `Field.declared_offset`; layout witness + generation; `commit()`
  propagation to embedders and cached array types; `_layout_is_fresh()` / `_ensure_layout()`.
* `dissect/cstruct/compiler.py` — generation-aware `_Fallback`; `_ensure_layout()` guard at the top of generated read
  and write methods; `_struct_read` global for the in-flight dispatch to the interpreted path.
* `dissect/cstruct/cstruct.py` — `_layout_version` counter.

The eight failing tests are the exact-source assertions listed above; they should be updated as part of any patch that
adopts this design. Nothing else in the suite changes behaviour.
