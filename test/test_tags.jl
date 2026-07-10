using ITensorBase: Index, IndexName, decoration, emptytags, gettag, gettags, hastag, inds,
    name, plev, prime, settags, tags, uniquename, unsettags
using MatrixAlgebraKit: eigh_full, left_null, qr_compact, svd_compact
using Test: @test, @test_throws, @testset

# The new leg of a factor, i.e. the one that is not among the kept indices.
newbond(t, keep) = only(filter(!in(keep), collect(inds(t))))

@testset "Tags and index decoration" begin
    @testset "IndexName tag queries" begin
        n = uniquename(IndexName; tags = "a" => "1", plev = 2)
        @test tags(n) == Dict("a" => "1")
        @test plev(n) == 2
        @test hastag(n, "a")
        @test !hastag(n, "b")
        @test gettag(n, "a") == "1"
        @test gettag(n, "b", "def") == "def"
        @test_throws KeyError gettag(n, "b")
    end

    @testset "settags is a merge over several input forms" begin
        i = Index(2)
        @test tags(settags(i, "a" => "1")) == Dict("a" => "1")                     # single pair
        @test tags(settags(i, "a" => "1", "b" => "2")) == Dict("a" => "1", "b" => "2") # varargs
        @test tags(settags(i, ["a" => "1", "b" => "2"])) == Dict("a" => "1", "b" => "2") # collection
        @test tags(settags(i, Dict("a" => "1"))) == Dict("a" => "1")               # AbstractDict
        @test tags(settags(i, :a => :b)) == Dict("a" => "b")                       # Symbols
        # merge: other keys kept, existing key overwritten
        j = settags(i, "a" => "1", "b" => "2")
        @test tags(settags(j, "a" => "9", "c" => "3")) ==
            Dict("a" => "9", "b" => "2", "c" => "3")
    end

    @testset "unsettags / emptytags are permissive" begin
        i = settags(Index(2), "a" => "1", "b" => "2", "c" => "3")
        @test tags(unsettags(i, ["a", "zzz"])) == Dict("b" => "2", "c" => "3")  # absent key ignored
        @test tags(unsettags(i, ("a", "b"))) == Dict("c" => "3")
        @test isempty(tags(emptytags(i)))
    end

    @testset "gettags returns the present subset" begin
        i = settags(Index(2), "a" => "1", "b" => "2")
        @test gettags(i, ["a", "b"]) == Dict("a" => "1", "b" => "2")
        @test gettags(i, ["a", "zzz"]) == Dict("a" => "1")  # absent key skipped
        @test isempty(gettags(i, ["zzz"]))
    end

    @testset "Index keyword constructor" begin
        i = Index(2; tags = "i" => "1", plev = 1)
        @test length(i) == 2
        @test tags(i) == Dict("i" => "1")
        @test plev(i) == 1
        @test isempty(tags(Index(2)))                              # bare default unchanged
        @test Index(2) != Index(2)                                 # fresh unique names
        @test Index(2; tags = "i" => "1") != Index(2; tags = "i" => "1")
    end

    @testset "decoration round-trips through uniquename" begin
        i = Index(2; tags = "Link" => "1", plev = 2)
        @test decoration(i) == (; tags = Dict("Link" => "1"), plev = 2)
        k = uniquename(IndexName; decoration(i)...)
        @test tags(k) == Dict("Link" => "1")
        @test plev(k) == 2
        @test k != name(i)                                         # fresh id
        @test decoration("x") == (;)                               # undecorated name
    end

    @testset "factorization bond-name decoration" begin
        i = Index(2)
        j = Index(3)
        a = randn(2, 3)[i, j]

        # Single new bond: `name`; default mints a bare bond.
        q, r = qr_compact(a, (i,), (j,); name = (; tags = "Link" => "1"))
        @test tags(newbond(q, (i,))) == Dict("Link" => "1")
        @test isempty(tags(newbond(qr_compact(a, (i,), (j,))[1], (i,))))

        # SVD: `leftname` (U-side) and `rightname` (V-side); S carries both legs.
        u, s, v = svd_compact(
            a, (i,), (j,); leftname = (; tags = "u" => "1"),
            rightname = (; tags = "v" => "1", plev = 1)
        )
        ub = newbond(u, (i,))
        vb = newbond(v, (j,))
        @test tags(ub) == Dict("u" => "1")
        @test tags(vb) == Dict("v" => "1")
        @test plev(vb) == 1
        @test Set(collect(inds(s))) == Set([ub, vb])

        # A callable spec gets full control over minting.
        u2, _, _ =
            svd_compact(a, (i,), (j,); leftname = nt -> uniquename(nt; tags = "c" => "9"))
        @test tags(newbond(u2, (i,))) == Dict("c" => "9")

        # `decoration` forwards an existing index's decoration onto a fresh bond.
        src = Index(2; tags = "Link" => "7", plev = 2)
        u3, _, _ = svd_compact(a, (i,), (j,); leftname = decoration(src))
        @test tags(newbond(u3, (i,))) == Dict("Link" => "7")
        @test plev(newbond(u3, (i,))) == 2

        # Eigen: D's legs are left/right (independent); V shares the right leg.
        s1 = Index(2)
        s1p = prime(s1)
        m = randn(2, 2)
        m = m + transpose(m)
        h = m[s1, s1p]
        d, vec = eigh_full(
            h, (s1,), (s1p,); leftname = (; tags = "row" => "1"),
            rightname = (; tags = "col" => "1")
        )
        @test Set(Dict.(tags.(collect(inds(d))))) ==
            Set([Dict("row" => "1"), Dict("col" => "1")])
        @test tags(newbond(vec, (s1p,))) == Dict("col" => "1")

        # Single new bond on the null spaces.
        n = left_null(a, (i,), (j,); name = (; tags = "n" => "1"))
        @test tags(newbond(n, (i,))) == Dict("n" => "1")
    end
end
