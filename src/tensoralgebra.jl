using LinearAlgebra: LinearAlgebra as LA
using MatrixAlgebraKit: MatrixAlgebraKit as MAK
using TensorAlgebra.MatrixAlgebra: MatrixAlgebra as MA
using TensorAlgebra: TensorAlgebra as TA

# This layer is used to define derivative rules (to skip differentiating `setdiff`).
names_setdiff(s1, s2) = setdiff(s1, s2)

Base.:*(a1::AbstractNamedTensor, a2::AbstractNamedTensor) = mul_nameddims(a1, a2)
function mul_nameddims(a1::AbstractNamedTensor, a2::AbstractNamedTensor)
    a_dest, names_dest = TA.contract(
        unnamed(a1), names(a1), unnamed(a2), names(a2)
    )
    return NamedTensor(a_dest, names_dest)
end

# Left associative fold/reduction.
# Circumvent Base definitions:
# ```julia
# *(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
# *(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
# ```
# that optimize matrix multiplication sequence.
function Base.:*(
        a1::AbstractNamedTensor, a2::AbstractNamedTensor,
        a3::AbstractNamedTensor, a_rest::AbstractNamedTensor...
    )
    return mul_nameddims(a1, a2, a3, a_rest...)
end
function mul_nameddims(
        a1::AbstractNamedTensor, a2::AbstractNamedTensor,
        a3::AbstractNamedTensor, a_rest::AbstractNamedTensor...
    )
    return *(*(a1, a2), a3, a_rest...)
end

function LA.mul!(
        a_dest::AbstractNamedTensor,
        a1::AbstractNamedTensor, a2::AbstractNamedTensor,
        α::Number, β::Number
    )
    return mul!_nameddims(a_dest, a1, a2, α, β)
end
function mul!_nameddims(
        a_dest::AbstractNamedTensor,
        a1::AbstractNamedTensor, a2::AbstractNamedTensor,
        α::Number, β::Number
    )
    TA.contractadd!(
        unnamed(a_dest), names(a_dest),
        unnamed(a1), names(a1),
        unnamed(a2), names(a2),
        α, β
    )
    return a_dest
end

function LA.mul!(
        a_dest::AbstractNamedTensor,
        a1::AbstractNamedTensor, a2::AbstractNamedTensor
    )
    return mul!_nameddims(a_dest, a1, a2)
end
function mul!_nameddims(
        a_dest::AbstractNamedTensor,
        a1::AbstractNamedTensor, a2::AbstractNamedTensor
    )
    TA.contract!(
        unnamed(a_dest), names(a_dest),
        unnamed(a1), names(a1),
        unnamed(a2), names(a2)
    )
    return a_dest
end

# Locate the named-dimension groups `group1`, `group2` within `a`, returning their two
# positional index groups.
function nameperm(a::AbstractNamedTensor, group1, group2)
    return TA.biperm(names(a), name.(Tuple(group1)), name.(Tuple(group2)))
end

"""
    TensorAlgebra.matricize(a::AbstractNamedTensor, codomain, domain)

Reshape the named tensor `a` into an unnamed matrix, fusing the `codomain` dimension group
into the rows and the `domain` group into the columns. `codomain` and `domain` are each any
iterable of dimensions (or dimension names) of `a`, and together they must cover all of `a`'s
dimensions.

# Examples

```jldoctest
julia> using TensorAlgebra: matricize

julia> i, j, k, l = Index.((2, 3, 2, 3));

julia> a = randn(i, j, k, l);

julia> size(matricize(a, (i, k), (j, l)))
(4, 9)
```
"""
function TA.matricize(a::AbstractNamedTensor, codomain, domain)
    perm_codomain, perm_domain = nameperm(a, codomain, domain)
    return TA.matricize(unnamed(a), perm_codomain, perm_domain)
end

# Unmatricize an unnamed matrix into the named `codomain`/`domain` axes, giving a named tensor.
# `Tuple{Vararg{NamedUnitRange}}` also matches an empty tuple, so demanding at least one named
# axis across the two groups takes three methods: one per group, plus the both-nonempty case
# that resolves the ambiguity between them.
function TA.unmatricize(
        m,
        codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
        domain::Tuple{Vararg{NamedUnitRange}}
    )
    return unmatricize_nameddims(m, codomain, domain)
end
function TA.unmatricize(
        m,
        codomain::Tuple{Vararg{NamedUnitRange}},
        domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return unmatricize_nameddims(m, codomain, domain)
end
function TA.unmatricize(
        m,
        codomain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
        domain::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}
    )
    return unmatricize_nameddims(m, codomain, domain)
end
function unmatricize_nameddims(m, codomain, domain)
    a = TA.unmatricize(m, space.(codomain), space.(domain))
    return NamedTensor(a, name.(codomain), name.(domain))
end

"""
    directsum(A => inds_A, B => inds_B, ...)
    directsum(out_inds, A => inds_A, B => inds_B, ...)

Direct sum of the named tensors `A, B, …` over the indices paired with each. The remaining
("shared") indices are common to every tensor; they are aligned and carried through unchanged,
while the paired indices are concatenated block-diagonally. The result has the shared indices
first and the summed indices trailing.

The first form mints fresh summed indices and returns `S => out_inds`, mirroring the
`tensor => indices` inputs. The second form takes the summed indices' names from `out_inds`
(names or `NamedUnitRange`s) and returns just `S`; the summed axes themselves come from the
direct sum, so only the names of `out_inds` are used.

# Examples

```jldoctest
julia> using ITensorBase: Index

julia> using TensorAlgebra: directsum

julia> i, j, k = Index.((2, 2, 3));

julia> a = randn(i, j);

julia> b = randn(i, k);

julia> s, (l,) = directsum(a => (j,), b => (k,));

julia> length(l)
5
```
"""
function TA.directsum(
        out_inds, pair1::Pair{<:AbstractNamedTensor}, pairs::Pair{<:AbstractNamedTensor}...
    )
    ps = (pair1, pairs...)
    shared = namesetdiff(inds(first(pair1)), last(pair1))
    summed_dims = length(shared) .+ eachindex(last(pair1))
    aligned_arrays = map(p -> unname(first(p), [shared; collect(last(p))]), ps)
    a = TA.directsum(summed_dims, aligned_arrays...)
    return NamedTensor(a, [name.(shared); name.(collect(out_inds))])
end
function TA.directsum(
        pair1::Pair{<:AbstractNamedTensor}, pairs::Pair{<:AbstractNamedTensor}...
    )
    out_names = [uniquename(nametype(first(pair1))) for _ in last(pair1)]
    s = TA.directsum(out_names, pair1, pairs...)
    return s => last(inds(s), length(out_names))
end

# Canonicalize a bond-name keyword to a `nametype -> name` minting function. A `NamedTuple` of
# decoration is splatted into `uniquename` (kwargs such as `tags`/`plev`); a callable is used
# as-is, for full control over how the name is minted. The default `(;)` reproduces the bare
# `uniquename(nametype)`. Each factorization exposes one such keyword per new bond it mints:
# `name` for a single shared bond, and `leftname`/`rightname` for the two legs of the central
# matrix in SVD and eigendecomposition.
function to_uniquename_function(decoration::NamedTuple)
    return nametype -> uniquename(nametype; decoration...)
end
to_uniquename_function(f) = f

for f in [
        :left_orth, :left_polar, :lq_compact, :lq_full, :qr_compact, :qr_full,
        :right_orth, :right_polar,
    ]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function MAK.$f(
                a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
            )
            return $f_nameddims(a, names_codomain, names_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, names_codomain, names_domain;
                name = (;), kwargs...
            )
            # `name` is a keyword here, so reach the `name` function through the module.
            codomain = ITensorBase.name.(names_codomain)
            domain = ITensorBase.name.(names_domain)
            x_unnamed, y_unnamed =
                TA.$f(unnamed(a), names(a), codomain, domain; kwargs...)
            name_x = to_uniquename_function(name)(nametype(a))
            name_y = name_x
            names_x = (codomain..., name_x)
            names_y = (name_y, domain...)
            x = NamedTensor(x_unnamed, names_x)
            y = NamedTensor(y_unnamed, names_y)
            return x, y
        end
        function MAK.$f(a::AbstractNamedTensor, names_codomain; kwargs...)
            return $f_nameddims(a, names_codomain; kwargs...)
        end
        function $f_nameddims(a::AbstractNamedTensor, names_codomain; kwargs...)
            codomain = name.(names_codomain)
            domain = names_setdiff(names(a), codomain)
            return MAK.$f(a, codomain, domain; kwargs...)
        end
    end
end

#
# SVD (three-output).
#

for f in [:svd_compact, :svd_full]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function MAK.$f(
                a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
            )
            return $f_nameddims(a, names_codomain, names_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, names_codomain, names_domain;
                leftname = (;), rightname = (;), kwargs...
            )
            codomain = name.(names_codomain)
            domain = name.(names_domain)
            u_unnamed, s_unnamed, v_unnamed = TA.$f(
                unnamed(a), names(a), codomain, domain; kwargs...
            )
            name_u = to_uniquename_function(leftname)(nametype(a))
            name_v = to_uniquename_function(rightname)(nametype(a))
            names_u = (codomain..., name_u)
            names_s = (name_u, name_v)
            names_v = (name_v, domain...)
            u = NamedTensor(u_unnamed, names_u)
            s = NamedTensor(s_unnamed, names_s)
            v = NamedTensor(v_unnamed, names_v)
            return u, s, v
        end
        function MAK.$f(a::AbstractNamedTensor, names_codomain; kwargs...)
            return $f_nameddims(a, names_codomain; kwargs...)
        end
        function $f_nameddims(a::AbstractNamedTensor, names_codomain; kwargs...)
            return MAK.$f(
                a,
                names_codomain,
                names_setdiff(names(a), name.(names_codomain));
                kwargs...
            )
        end
    end
end

# `svd_trunc` mirrors the three-output SVD above but also returns the truncation error `ϵ`
# (the 2-norm of the discarded singular values), matching MatrixAlgebraKit's four-output
# `svd_trunc`, so it is spelled out here rather than sharing the loop.
function MAK.svd_trunc(
        a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
    )
    return svd_trunc_nameddims(a, names_codomain, names_domain; kwargs...)
end
function svd_trunc_nameddims(
        a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
    )
    codomain = name.(names_codomain)
    domain = name.(names_domain)
    u_unnamed, s_unnamed, v_unnamed, ϵ = TA.svd_trunc(
        unnamed(a), names(a), codomain, domain; kwargs...
    )
    name_u = uniquename(nametype(a))
    name_v = uniquename(nametype(a))
    u = NamedTensor(u_unnamed, (codomain..., name_u))
    s = NamedTensor(s_unnamed, (name_u, name_v))
    v = NamedTensor(v_unnamed, (name_v, domain...))
    return u, s, v, ϵ
end
function MAK.svd_trunc(a::AbstractNamedTensor, names_codomain; kwargs...)
    return svd_trunc_nameddims(a, names_codomain; kwargs...)
end
function svd_trunc_nameddims(a::AbstractNamedTensor, names_codomain; kwargs...)
    return MAK.svd_trunc(
        a,
        names_codomain,
        names_setdiff(names(a), name.(names_codomain));
        kwargs...
    )
end

#
# Singular values.
#

function MAK.svd_vals(
        a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
    )
    return svd_vals_nameddims(a, names_codomain, names_domain; kwargs...)
end
function svd_vals_nameddims(
        a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
    )
    return TA.svd_vals(
        unnamed(a),
        names(a),
        name.(names_codomain),
        name.(names_domain);
        kwargs...
    )
end

function MAK.svd_vals(a::AbstractNamedTensor, names_codomain; kwargs...)
    return svd_vals_nameddims(a, names_codomain; kwargs...)
end
function svd_vals_nameddims(a::AbstractNamedTensor, names_codomain; kwargs...)
    codomain = name.(names_codomain)
    domain = names_setdiff(names(a), codomain)
    return MAK.svd_vals(a, codomain, domain; kwargs...)
end

#
# Eigendecomposition (two-output).
#

for f in [:eigh_full, :eig_full, :eigh_trunc, :eig_trunc]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function MAK.$f(
                a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
            )
            return $f_nameddims(a, names_codomain, names_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, names_codomain, names_domain;
                leftname = (;), rightname = (;), kwargs...
            )
            codomain = name.(names_codomain)
            domain = name.(names_domain)
            d_unnamed, v_unnamed = TA.$f(
                unnamed(a), names(a), codomain, domain; kwargs...
            )
            name_d = to_uniquename_function(rightname)(nametype(a))
            name_d′ = to_uniquename_function(leftname)(nametype(a))
            name_v = name_d
            names_d = (name_d′, name_d)
            names_v = (domain..., name_v)
            d = NamedTensor(d_unnamed, names_d)
            v = NamedTensor(v_unnamed, names_v)
            return d, v
        end
    end
end

#
# Eigenvalues.
#

for f in [:eigh_vals, :eig_vals]
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function MAK.$f(
                a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
            )
            return $f_nameddims(a, names_codomain, names_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
            )
            codomain = name.(names_codomain)
            domain = name.(names_domain)
            return TA.$f(unnamed(a), names(a), codomain, domain; kwargs...)
        end
    end
end

function MAK.left_null(
        a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
    )
    return left_null_nameddims(a, names_codomain, names_domain; kwargs...)
end
function left_null_nameddims(
        a::AbstractNamedTensor, names_codomain, names_domain; name = (;), kwargs...
    )
    # `name` is a keyword here, so reach the `name` function through the module.
    codomain = ITensorBase.name.(names_codomain)
    domain = ITensorBase.name.(names_domain)
    n_unnamed = TA.left_null(unnamed(a), names(a), codomain, domain; kwargs...)
    name_n = to_uniquename_function(name)(nametype(a))
    names_n = (codomain..., name_n)
    return NamedTensor(n_unnamed, names_n)
end

function MAK.left_null(a::AbstractNamedTensor, names_codomain; kwargs...)
    return left_null_nameddims(a, names_codomain; kwargs...)
end
function left_null_nameddims(a::AbstractNamedTensor, names_codomain; kwargs...)
    codomain = name.(names_codomain)
    domain = names_setdiff(names(a), codomain)
    return MAK.left_null(a, codomain, domain; kwargs...)
end

function MAK.right_null(
        a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
    )
    return right_null_nameddims(a, names_codomain, names_domain; kwargs...)
end
function right_null_nameddims(
        a::AbstractNamedTensor, names_codomain, names_domain; name = (;), kwargs...
    )
    # `name` is a keyword here, so reach the `name` function through the module.
    codomain = ITensorBase.name.(names_codomain)
    domain = ITensorBase.name.(names_domain)
    n_unnamed = TA.right_null(unnamed(a), names(a), codomain, domain; kwargs...)
    name_n = to_uniquename_function(name)(nametype(a))
    names_n = (name_n, domain...)
    return NamedTensor(n_unnamed, names_n)
end

function MAK.right_null(a::AbstractNamedTensor, names_codomain; kwargs...)
    return right_null_nameddims(a, names_codomain; kwargs...)
end
function right_null_nameddims(a::AbstractNamedTensor, names_codomain; kwargs...)
    codomain = name.(names_codomain)
    domain = names_setdiff(names(a), codomain)
    return MAK.right_null(a, codomain, domain; kwargs...)
end

"""
    TensorAlgebra.MatrixAlgebra.sqrth_safe(a::AbstractNamedTensor, names_codomain, names_domain; kwargs...) -> p

Square root of a named array `a`, interpreting it as a Hermitian positive
semi-definite linear map from the domain to the codomain dimension names.
The result carries the same dimension names as `a`. Eigenvalues below
tolerance are clamped to zero. The input must be Hermitian: project with
`MatrixAlgebraKit.project_hermitian` first if it is Hermitian only up to
numerical noise.

`kwargs` are forwarded to `TensorAlgebra.sqrth_safe` on the underlying
unnamed array (e.g. `atol`, `rtol`).

See also [`TensorAlgebra.MatrixAlgebra.invsqrth_safe`](@ref) and
[`TensorAlgebra.MatrixAlgebra.sqrth_invsqrth_safe`](@ref).
"""
MA.sqrth_safe

"""
    TensorAlgebra.MatrixAlgebra.invsqrth_safe(a::AbstractNamedTensor, names_codomain, names_domain; kwargs...) -> p

Pseudo-inverse square root of a named array `a`, interpreting it as a
Hermitian positive semi-definite linear map from the domain to the codomain
dimension names. The result carries the same dimension names as `a`.
Eigenvalues below tolerance are clamped to zero (Moore-Penrose convention).
The input must be Hermitian: project with
`MatrixAlgebraKit.project_hermitian` first if it is Hermitian only up to
numerical noise.

`kwargs` are forwarded to `TensorAlgebra.invsqrth_safe` on the underlying
unnamed array (e.g. `atol`, `rtol`).

See also [`TensorAlgebra.MatrixAlgebra.sqrth_safe`](@ref) and
[`TensorAlgebra.MatrixAlgebra.sqrth_invsqrth_safe`](@ref).
"""
MA.invsqrth_safe

"""
    TensorAlgebra.MatrixAlgebra.sqrth_invsqrth_safe(a::AbstractNamedTensor, names_codomain, names_domain; kwargs...) -> p, pinv

Square root and pseudo-inverse square root of a named array `a` (see
`TensorAlgebra.MatrixAlgebra.sqrth_safe` and
`TensorAlgebra.MatrixAlgebra.invsqrth_safe`), from a single
eigendecomposition. Both results carry the same dimension names as `a`.

`kwargs` are forwarded to `TensorAlgebra.sqrth_invsqrth_safe` on the underlying
unnamed array (e.g. `atol`, `rtol`).
"""
MA.sqrth_invsqrth_safe

"""
    MatrixAlgebraKit.project_hermitian(a::AbstractNamedTensor, names_codomain, names_domain; kwargs...) -> h

Hermitian part `(m + m') / 2` of a named array `a`, interpreting it as a
linear map `m` from the domain to the codomain dimension names. The result
carries the same dimension names as `a`.
"""
MAK.project_hermitian

# The named forms above: lower to the corresponding tensor-level TensorAlgebra function on
# the unnamed array and reattach the names, codomain first. `sqrth_invsqrth_safe` differs
# only in fanning the names out over its result pair.
for (M, f) in ((MA, :sqrth_safe), (MA, :invsqrth_safe), (MAK, :project_hermitian))
    @eval function $M.$f(
            a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
        )
        codomain = name.(names_codomain)
        domain = name.(names_domain)
        p_unnamed = TA.$f(unnamed(a), names(a), codomain, domain; kwargs...)
        return NamedTensor(p_unnamed, (codomain..., domain...))
    end
end
function MA.sqrth_invsqrth_safe(
        a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
    )
    codomain = name.(names_codomain)
    domain = name.(names_domain)
    p_unnamed, pinv_unnamed = TA.sqrth_invsqrth_safe(
        unnamed(a), names(a), codomain, domain; kwargs...
    )
    names_p = (codomain..., domain...)
    return NamedTensor(p_unnamed, names_p), NamedTensor(pinv_unnamed, names_p)
end

"""
    Base.one(a::AbstractNamedTensor, names_codomain, names_domain) -> Id

Return an identity-operator-shaped named array sharing `a`'s dimension names,
codomain/domain partition, and element type. The fused codomain and domain sizes
must match. `a` is treated as a shape prototype and is not mutated.

The identity acts as the multiplicative identity for `ITensorBase.apply`: it
contracts on the domain names and renames the resulting codomain names back to
the domain names, leaving the input unchanged.

Note that this is inspired by the tensor map function `TensorKit.one` in
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl).

# Examples

```jldoctest
julia> using ITensorBase: Index

julia> using LinearAlgebra: tr

julia> i, j, k, l = Index.((2, 3, 2, 3));

julia> a = randn(i, j, k, l);

julia> tr(one(a, (i, j), (k, l)), (i, j), (k, l))
6.0
```
"""
function Base.one(
        a::AbstractNamedTensor, names_codomain, names_domain
    )
    return one_nameddims(a, names_codomain, names_domain)
end
function one_nameddims(
        a::AbstractNamedTensor, names_codomain, names_domain
    )
    codomain = name.(names_codomain)
    domain = name.(names_domain)
    raw = TA.one(unnamed(a), names(a), codomain, domain)
    return NamedTensor(raw, (codomain..., domain...))
end

"""
    id(elt::Type, codomain, domain) -> Id

Construct a from-scratch identity-operator-shaped named tensor over the `codomain` and
`domain` indices, with element type `elt`. The fused codomain and domain sizes must match.
Unlike [`one`](@ref), which follows a prototype tensor, `id` needs only the indices and an
element type, so it is the right primitive when no prototype is in hand. The index axes select
the backend: dense ranges give a dense tensor, graded ranges a block-sparse one.

Note that this is inspired by the tensor map function `TensorKit.id` in
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl).

# Examples

```jldoctest
julia> using ITensorBase: Index, id

julia> using LinearAlgebra: tr

julia> i, j, k, l = Index.((2, 3, 2, 3));

julia> tr(id(Float64, (i, j), (k, l)), (i, j), (k, l))
6.0
```

See also [`one`](@ref).
"""
function id(elt::Type, codomain, domain)
    codomain, domain = Tuple(codomain), Tuple(domain)
    m = Matrix{elt}(LA.I, prod(length, codomain), prod(length, domain))
    axissizes = (length.(codomain)..., length.(domain)...)
    return TA.project(reshape(m, axissizes), codomain, domain)
end

"""
    LinearAlgebra.tr(a::AbstractNamedTensor, codomain, domain) -> scalar

Trace of `a` viewed as a map, pairing each `codomain` index with the `domain` index in the
same position (matching sizes) and summing the diagonal. `codomain` and `domain` together
must cover all of `a`'s indices, so the result is a scalar. Forwards to `TensorAlgebra.tr` on
the unnamed data, which matricizes `a` into its square matrix and takes the matrix trace, so it
follows `a`'s backend (dense, graded, or `TensorMap`).

# Examples

```jldoctest
julia> using ITensorBase: Index

julia> using LinearAlgebra: tr

julia> i, j, k, l = Index.((2, 3, 2, 3));

julia> tr(fill(2.0, (i, j, k, l)), (i, j), (k, l))
12.0
```
"""
function LA.tr(a::AbstractNamedTensor, codomain, domain)
    codomain, domain = Tuple(codomain), Tuple(domain)
    return TA.tr(unnamed(a), names(a), name.(codomain), name.(domain))
end

const MATRIX_FUNCTIONS = [
    :exp, :cis, :log, :sqrt, :cbrt, :cos, :sin, :tan, :csc, :sec, :cot, :cosh, :sinh,
    :tanh,
    :csch, :sech, :coth, :acos, :asin, :atan, :acsc, :asec, :acot, :acosh, :asinh,
    :atanh,
    :acsch, :asech, :acoth,
]

for f in MATRIX_FUNCTIONS
    f_nameddims = Symbol(f, "_nameddims")
    @eval begin
        function Base.$f(
                a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
            )
            return $f_nameddims(a, names_codomain, names_domain; kwargs...)
        end
        function $f_nameddims(
                a::AbstractNamedTensor, names_codomain, names_domain; kwargs...
            )
            codomain = name.(names_codomain)
            domain = name.(names_domain)
            fa_unnamed = TA.$f(
                unnamed(a), names(a), codomain, domain; kwargs...
            )
            return NamedTensor(fa_unnamed, (codomain..., domain...))
        end
    end
end

#
# Projection into a symmetry-restricted named tensor.
#

# Attach `input_names` to the projected array's axes, minting a fresh unique name for each trailing
# axis beyond them. The strict `project` verbs add none, so this just reattaches the given names;
# the `*_aux` verbs append one derived auxiliary leg (as the last domain axis) carrying the flux of
# a charge-shifting operator or non-invariant state, and naming it returns it as a dimension the
# caller can read off the result. A `nothing` (from the nullable verbs) passes straight through.
function name_projected(projected, input_names)
    isnothing(projected) && return nothing
    aux_names = ntuple(
        _ -> uniquename(eltype(input_names)),
        TA.ndims(projected) - length(input_names)
    )
    return NamedTensor(projected, (input_names..., aux_names...))
end

# Each `<verb>_nameddims` runs the named-index layer of a `TensorAlgebra` verb: strip the axes to
# their unnamed ranges, lower to the unnamed verb, and reattach the names (the `*_aux` verbs also
# name the derived auxiliary leg, see `name_projected`). The one body also covers an empty codomain,
# since `unnamed.(())` and `name.(())` are both `()`, so the all-domain (co-state) case needs no
# separate path.
function project_nameddims(a, codomain_inds, domain_inds; kwargs...)
    projected = TA.project(a, unnamed.(codomain_inds), unnamed.(domain_inds); kwargs...)
    return name_projected(projected, (name.(codomain_inds)..., name.(domain_inds)...))
end
function tryproject_nameddims(a, codomain_inds, domain_inds; kwargs...)
    projected = TA.tryproject(a, unnamed.(codomain_inds), unnamed.(domain_inds); kwargs...)
    return name_projected(projected, (name.(codomain_inds)..., name.(domain_inds)...))
end
function unchecked_project_nameddims(a, codomain_inds, domain_inds; kwargs...)
    projected =
        TA.unchecked_project(a, unnamed.(codomain_inds), unnamed.(domain_inds); kwargs...)
    return name_projected(projected, (name.(codomain_inds)..., name.(domain_inds)...))
end

# The `*_aux` workers derive and append the flux-carrying auxiliary leg, which `name_projected`
# names. Same named-index layer as the strict workers above, lowered to the `*_aux` unnamed verbs.
function project_aux_nameddims(a, codomain_inds, domain_inds; kwargs...)
    projected = TA.project_aux(a, unnamed.(codomain_inds), unnamed.(domain_inds); kwargs...)
    return name_projected(projected, (name.(codomain_inds)..., name.(domain_inds)...))
end
function tryproject_aux_nameddims(a, codomain_inds, domain_inds; kwargs...)
    projected =
        TA.tryproject_aux(a, unnamed.(codomain_inds), unnamed.(domain_inds); kwargs...)
    return name_projected(projected, (name.(codomain_inds)..., name.(domain_inds)...))
end
function unchecked_project_aux_nameddims(a, codomain_inds, domain_inds; kwargs...)
    projected =
        TA.unchecked_project_aux(
        a,
        unnamed.(codomain_inds),
        unnamed.(domain_inds);
        kwargs...
    )
    return name_projected(projected, (name.(codomain_inds)..., name.(domain_inds)...))
end

# Shared body for the named-index `project` and `project_aux` family docstrings. Each function's
# summary states its own verification behavior; this describes the forms and backend selection they
# all share.
const _project_named_body = """
The three-argument form takes an explicit codomain/domain split (an operator), and the
two-argument form a flat list of indices (a state, i.e. an empty domain). The index axes select
the backend: dense ranges give an `Array`, graded ranges a block-sparse array, and TensorKit
spaces a `TensorMap`. `a` is indexed positionally in the order `(codomain_inds..., domain_inds...)`.
"""

# Extra paragraph for the `*_aux` docstrings: unlike `project`, which projects into exactly the given
# indices, these derive and append the flux-carrying leg.
const _project_aux_named_body = """
`a` may carry the physical rank the indices account for, or one trailing slice axis. `project_aux`
derives an auxiliary domain index to make the result symmetry-allowed (for example a flux-canceling
charge leg for a charge-shifting operator) and returns it as a named dimension with a freshly
generated name the caller can read off the result.
"""

const _project_named_docstring = """
    TensorAlgebra.project(a::AbstractArray, codomain_inds, domain_inds; kwargs...) -> t
    TensorAlgebra.project(a::AbstractArray, inds; kwargs...) -> t

Build a named tensor by projecting the dense array `a` into the symmetry-restricted space
described by the indices, verifying that only a negligible component of `a` is discarded and
throwing an `InexactError` otherwise (keyword arguments are forwarded to the `isapprox` tolerance
check).

$(_project_named_body)
`project` projects into exactly the given indices. To append a derived flux-carrying leg for a
charge-shifting operator or non-invariant state, use `TensorAlgebra.project_aux`.

See also `TensorAlgebra.tryproject` and `TensorAlgebra.unchecked_project`.
"""

const _tryproject_named_docstring = """
    TensorAlgebra.tryproject(a::AbstractArray, codomain_inds, domain_inds; kwargs...) -> Union{t, Nothing}
    TensorAlgebra.tryproject(a::AbstractArray, inds; kwargs...) -> Union{t, Nothing}

Like `TensorAlgebra.project`, but return `nothing` instead of throwing when a non-negligible
component of `a` would be discarded (keyword arguments are forwarded to the `isapprox` tolerance
check).

$(_project_named_body)
See also `TensorAlgebra.project`, `TensorAlgebra.unchecked_project`, and `TensorAlgebra.tryproject_aux`.
"""

const _unchecked_project_named_docstring = """
    TensorAlgebra.unchecked_project(a::AbstractArray, codomain_inds, domain_inds; kwargs...) -> t
    TensorAlgebra.unchecked_project(a::AbstractArray, inds; kwargs...) -> t

Like `TensorAlgebra.project`, but skip the verification: components of `a` outside the
symmetry-allowed structure are dropped without inspection.

$(_project_named_body)
See also `TensorAlgebra.project`, `TensorAlgebra.tryproject`, and `TensorAlgebra.unchecked_project_aux`.
"""

const _project_aux_named_docstring = """
    TensorAlgebra.project_aux(a::AbstractArray, codomain_inds, domain_inds; kwargs...) -> t
    TensorAlgebra.project_aux(a::AbstractArray, inds; kwargs...) -> t

Build a named tensor by projecting `a` and appending a derived auxiliary domain index carrying its
flux, verifying that only a negligible component of `a` is discarded and throwing an `InexactError`
otherwise (keyword arguments are forwarded to the `isapprox` tolerance check).

$(_project_named_body)
$(_project_aux_named_body)
See also `TensorAlgebra.tryproject_aux` and `TensorAlgebra.unchecked_project_aux`.
"""

const _tryproject_aux_named_docstring = """
    TensorAlgebra.tryproject_aux(a::AbstractArray, codomain_inds, domain_inds; kwargs...) -> Union{t, Nothing}
    TensorAlgebra.tryproject_aux(a::AbstractArray, inds; kwargs...) -> Union{t, Nothing}

Like `TensorAlgebra.project_aux`, but return `nothing` instead of throwing when a non-negligible
component of `a` would be discarded (keyword arguments are forwarded to the `isapprox` tolerance
check).

$(_project_named_body)
$(_project_aux_named_body)
See also `TensorAlgebra.project_aux` and `TensorAlgebra.unchecked_project_aux`.
"""

const _unchecked_project_aux_named_docstring = """
    TensorAlgebra.unchecked_project_aux(a::AbstractArray, codomain_inds, domain_inds; kwargs...) -> t
    TensorAlgebra.unchecked_project_aux(a::AbstractArray, inds; kwargs...) -> t

Like `TensorAlgebra.project_aux`, but skip the verification: components of `a` outside the
symmetry-allowed structure are dropped without inspection.

$(_project_named_body)
$(_project_aux_named_body)
See also `TensorAlgebra.project_aux` and `TensorAlgebra.tryproject_aux`.
"""

# Forward each named-index signature to its worker, attaching the family docstring to the split
# form. Two split entries per verb so an empty codomain or an empty domain still selects this
# overload instead of the unnamed-axis generic. The flat (state) form forwards with an empty domain.
for f in (
        :project, :tryproject, :unchecked_project,
        :project_aux, :tryproject_aux, :unchecked_project_aux,
    )
    fnamed = Symbol(f, :_nameddims)
    doc = Symbol("_", f, "_named_docstring")
    @eval begin
        @doc $doc function TA.$f(
                a::AbstractArray,
                codomain_inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}},
                domain_inds::Tuple{Vararg{NamedUnitRange}}; kwargs...
            )
            return $fnamed(a, codomain_inds, domain_inds; kwargs...)
        end
        function TA.$f(
                a::AbstractArray,
                codomain_inds::Tuple{},
                domain_inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}}; kwargs...
            )
            return $fnamed(a, codomain_inds, domain_inds; kwargs...)
        end
        function TA.$f(
                a::AbstractArray, inds::Tuple{NamedUnitRange, Vararg{NamedUnitRange}};
                kwargs...
            )
            return $fnamed(a, inds, (); kwargs...)
        end
    end
end
