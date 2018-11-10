using Base.Test
using BSBP

@testset "Simulation" begin
    score,gap,rtime,ncount = Simulation()
    @test mean(score) >= 7
    @test mean(gap) == 0
    @test mean(rtime) <= 2
    @test mean(ncount) <= 2
end

@testset "TransportationMode" begin
    score,gap,rtime,ncount = TransportationMode()
    @test score == 756
    @test gap <= 40
    @test rtime <= 100
    @test ncount >= 2000
end

@testset "TransportationModeCV" begin
    score,gap,rtime,ncount = TransportationModeCV()
    @test mean(score) >=605
    @test mean(gap) <= 40
    @test mean(rtime) <= 100
    @test mean(ncount) >= 2000
end
