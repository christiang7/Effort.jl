using Test
using Effort
using DifferentiationInterface
using ForwardDiff
using Zygote

@testset "Velocileptors Emulators" begin

    # Common test parameters (can use same as other tests)
    z_test = 0.8
    ln10As_test = 3.044
    ns_test = 0.9649
    H0_test = 67.36
    ωb_test = 0.02237
    ωcdm_test = 0.12
    mν_test = 0.06
    w0_test = -1.0
    wa_test = 0.0
    cosmology_params = [z_test, ln10As_test, ns_test, H0_test, ωb_test, ωcdm_test, mν_test, w0_test, wa_test]

    bias_params = [2.0, -0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 1.0, 1.0, 1.0]
    D_growth = 0.75

    @testset "VelocileptorsREPTmnuw0wacdm" begin
        emu_key = "VelocileptorsREPTmnuw0wacdm"
        @test haskey(Effort.trained_emulators, emu_key)
        
        monopole_emu = Effort.trained_emulators[emu_key]["0"]
        quadrupole_emu = Effort.trained_emulators[emu_key]["2"]
        hexadecapole_emu = Effort.trained_emulators[emu_key]["4"]

        @test !isnothing(monopole_emu)
        @test !isnothing(quadrupole_emu)
        @test !isnothing(hexadecapole_emu)

        @testset "Multipole Computation" begin
            P0 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, monopole_emu)
            P2 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, quadrupole_emu)
            P4 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, hexadecapole_emu)

            @test all(isfinite.(P0))
            @test all(isfinite.(P2))
            @test all(isfinite.(P4))
            @test length(P0) == length(P2) == length(P4)
        end

        @testset "Jacobian Computation" begin
            P0, J0 = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, monopole_emu)
            @test all(isfinite.(J0))
            @test size(J0, 1) == length(P0)
            @test size(J0, 2) == length(bias_params)
        end
        
        @testset "Zygote Differentiation" begin
             function loss(cosmo)
                P0 = Effort.get_Pℓ(cosmo, D_growth, bias_params, monopole_emu)
                return sum(P0)
             end
             grad = DifferentiationInterface.gradient(loss, AutoZygote(), cosmology_params)
             @test all(isfinite.(grad))
             @test length(grad) == length(cosmology_params)
        end

        @testset "Analytical vs AutoDiff Jacobian - Bias Parameters" begin
            # Compare get_Pℓ_jacobian vs ForwardDiff through get_Pℓ
            
            @testset "Monopole (ℓ=0)" begin
                P0_analytical, J0_analytical = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, monopole_emu)
                J0_autodiff = DifferentiationInterface.jacobian(
                    b -> Effort.get_Pℓ(cosmology_params, D_growth, b, monopole_emu),
                    AutoForwardDiff(),
                    bias_params
                )
                @test J0_autodiff ≈ J0_analytical rtol=1e-5
            end

            @testset "Quadrupole (ℓ=2)" begin
                P2_analytical, J2_analytical = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, quadrupole_emu)
                J2_autodiff = DifferentiationInterface.jacobian(
                    b -> Effort.get_Pℓ(cosmology_params, D_growth, b, quadrupole_emu),
                    AutoForwardDiff(),
                    bias_params
                )
                @test J2_autodiff ≈ J2_analytical rtol=1e-5
            end

            @testset "Hexadecapole (ℓ=4)" begin
                P4_analytical, J4_analytical = Effort.get_Pℓ_jacobian(cosmology_params, D_growth, bias_params, hexadecapole_emu)
                J4_autodiff = DifferentiationInterface.jacobian(
                    b -> Effort.get_Pℓ(cosmology_params, D_growth, b, hexadecapole_emu),
                    AutoForwardDiff(),
                    bias_params
                )
                @test J4_autodiff ≈ J4_analytical rtol=1e-5
            end
        end
    end

    @testset "VelocileptorsLPTmnuw0wacdm" begin
        emu_key = "VelocileptorsLPTmnuw0wacdm"
        # Check if loaded (might fail if artifact not installed/loaded in __init__ but we expect it to be there if defined)
        if haskey(Effort.trained_emulators, emu_key)
            monopole_emu = Effort.trained_emulators[emu_key]["0"]
            quadrupole_emu = Effort.trained_emulators[emu_key]["2"]
            hexadecapole_emu = Effort.trained_emulators[emu_key]["4"]

            @test !isnothing(monopole_emu)
            
            @testset "Multipole Computation" begin
                P0 = Effort.get_Pℓ(cosmology_params, D_growth, bias_params, monopole_emu)
                @test all(isfinite.(P0))
            end
             
             @testset "Zygote Differentiation" begin
                 function loss(cosmo)
                    P0 = Effort.get_Pℓ(cosmo, D_growth, bias_params, monopole_emu)
                    return sum(P0)
                 end
                 grad = DifferentiationInterface.gradient(loss, AutoZygote(), cosmology_params)
                 @test all(isfinite.(grad))
            end
        else
            @warn "VelocileptorsLPTmnuw0wacdm not found in trained_emulators. Skipping tests for it."
        end
    end

end