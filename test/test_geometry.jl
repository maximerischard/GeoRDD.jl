import LibGEOS
import GeoInterface

function squares()
    A_T = LibGEOS.readgeom("POLYGON((1 1,3 1,3 3,1 3,1 1))")
    A_C = LibGEOS.readgeom("POLYGON((3 1,5 1,5 3,3 3,3 1))")
    border = GeoRDD.get_border(A_T, A_C, 0.0)
    @test GeoInterface.coordinates(border) == [[3.0, 1.0], [3.0, 3.0]]
    border = GeoRDD.get_border(A_T, A_C, 1.0)
    @test GeoInterface.coordinates(border) == [[2.0, 1.0], [3.0, 1.0], [3.0, 3.0], [2.0, 3.0]]
end

function twosquaresT()
    A_T1 = LibGEOS.readgeom("POLYGON((3 1,5 1,5 3,3 3,3 1))")
    A_T2 = LibGEOS.readgeom("POLYGON((1 3,3 3,3 5,1 5,1 3))")
    A_T = LibGEOS.union(A_T1, A_T2)
    A_C = LibGEOS.readgeom("POLYGON((1 1,3 1,3 3,1 3,1 1))")
    border = GeoRDD.get_border(A_T, A_C, 0.0)
    @test GeoInterface.coordinates(border) == [[3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]
    border = GeoRDD.get_border(A_T, A_C, 1.0)
    # this isn't ideal:
    @test GeoInterface.coordinates(border) == [[[4.0, 1.0], [3.0, 1.0], [3.0, 3.0], [4.0, 3.0]], 
                                               [[3.0, 4.0], [3.0, 3.0], [1.0, 3.0], [1.0, 4.0]]]
end

function twosquaresC()
    A_C1 = LibGEOS.readgeom("POLYGON((3 1,5 1,5 3,3 3,3 1))")
    A_C2 = LibGEOS.readgeom("POLYGON((1 3,3 3,3 5,1 5,1 3))")
    A_C = LibGEOS.union(A_C1, A_C2)
    A_T = LibGEOS.readgeom("POLYGON((1 1,3 1,3 3,1 3,1 1))")
    border = GeoRDD.get_border(A_T, A_C, 0.0)
    @test GeoInterface.coordinates(border) == [[3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]
    border = GeoRDD.get_border(A_T, A_C, 1.0)
    # this one is better...
    @test GeoInterface.coordinates(border) == [[2.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 2.0]]
end

function twosquaresC_clock()
    A_C1 = LibGEOS.readgeom("POLYGON((3 1,5 1,5 3,3 3,3 1))")
    A_C2 = LibGEOS.readgeom("POLYGON((1 3,1 5,3 5,3 3,1 3))")
    A_C = LibGEOS.union(A_C1, A_C2)
    A_T = LibGEOS.readgeom("POLYGON((1 1,3 1,3 3,1 3,1 1))")
    border = GeoRDD.get_border(A_T, A_C, 0.0)
    @test GeoInterface.coordinates(border) == [[3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]
    border = GeoRDD.get_border(A_T, A_C, 1.0)
    # this one is better...
    @test GeoInterface.coordinates(border) == [[2.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0], [1.0, 2.0]]
end

function wrapsquares()
    A_T = LibGEOS.readgeom("POLYGON((3 1,5 1,5 5,1 5,1 3,3 3,3 1))")
    A_C = LibGEOS.readgeom("POLYGON((1 1,3 1,3 3,1 3,1 1))")
    border = GeoRDD.get_border(A_T, A_C, 0.0)
    @test GeoInterface.coordinates(border) == [[1.0, 3.0], [3.0, 3.0], [3.0, 1.0]]
end

@testset "test geometry" begin
    squares()
    twosquaresT()
    twosquaresC()
    twosquaresC_clock()
    wrapsquares()
end
