# Epidemic Model - Simplified Comparison of ReLU, Tanh, and Sigmoid
# Focus: Infected & Recovered predictions, Speed, and Errors only

import Pkg
Pkg.activate(@__DIR__)

if !isfile(joinpath(@__DIR__, "Project.toml"))
    Pkg.activate(@__DIR__)
    Pkg.add(["Statistics", "Lux", "DiffEqFlux", "DifferentialEquations", 
             "Optimization", "OptimizationOptimJL", "OptimizationOptimisers", 
             "Random", "Plots", "ComponentArrays", "Printf", "JLD2"])
end

using Lux, DiffEqFlux, DifferentialEquations, Optimization
using OptimizationOptimJL, OptimizationOptimisers, Random, Plots, ComponentArrays
using Statistics, Printf
using JLD2  # Add JLD2 for saving results

# ========== 1. Generate Data ==========
N_days = 25
const S0 = 1.0
u0 = [S0*0.99, S0*0.01, 0.0, 0.0, 0.0]

# True parameters
p0 = [0.85, 0.1, 0.05, 0.025, 0.02, 0.002]

tspan = (0.0, Float64(N_days))
t = range(tspan[1], tspan[2], length=N_days)

# True model
function SIRHD!(du, u, p, t)
    S, I, R, H, D = u
    τSI, τIR, τID, τIH, τHR, τHD = abs.(p)
    du[1] = -τSI * S * I
    du[2] = τSI * S * I - (τIR + τID + τIH) * I
    du[3] = τIR * I + τHR * H
    du[4] = τIH * I - (τHR + τHD) * H
    du[5] = τID * I + τHD * H
end

prob = ODEProblem(SIRHD!, u0, tspan, p0)
sol = Array(solve(prob, Tsit5(), saveat=t))

# Extract data for training
Infected_Data = sol[2, :]
Recovered_Data = sol[3, :]

# ========== 2. Function to Create Neural Network Model ==========
function create_epidemic_model(activation_fn, rng)
    NN1 = Lux.Chain(Lux.Dense(2, 10, activation_fn), Lux.Dense(10, 1))
    NN2 = Lux.Chain(Lux.Dense(1, 10, activation_fn), Lux.Dense(10, 1))
    NN3 = Lux.Chain(Lux.Dense(1, 10, activation_fn), Lux.Dense(10, 1))
    NN4 = Lux.Chain(Lux.Dense(1, 10, activation_fn), Lux.Dense(10, 1))
    NN5 = Lux.Chain(Lux.Dense(1, 10, activation_fn), Lux.Dense(10, 1))
    NN6 = Lux.Chain(Lux.Dense(1, 10, activation_fn), Lux.Dense(10, 1))
    
    p1, st1 = Lux.setup(rng, NN1)
    p2, st2 = Lux.setup(rng, NN2)
    p3, st3 = Lux.setup(rng, NN3)
    p4, st4 = Lux.setup(rng, NN4)
    p5, st5 = Lux.setup(rng, NN5)
    p6, st6 = Lux.setup(rng, NN6)
    
    p_vec = (layer_1 = p1, layer_2 = p2, layer_3 = p3, 
             layer_4 = p4, layer_5 = p5, layer_6 = p6)
    p_vec = ComponentArray(p_vec)
    
    states = (st1, st2, st3, st4, st5, st6)
    
    function dxdt_pred(du, u, p, t)
        S, I, R, H, D = u
        
        NNSI = abs(NN1([S, I], p.layer_1, states[1])[1][1])
        NNIR = abs(NN2([I], p.layer_2, states[2])[1][1])
        NNID = abs(NN3([I], p.layer_3, states[3])[1][1])
        NNIH = abs(NN4([I], p.layer_4, states[4])[1][1])
        NNHR = abs(NN5([H], p.layer_5, states[5])[1][1])
        NNHD = abs(NN6([H], p.layer_6, states[6])[1][1])
        
        du[1] = -NNSI * S * I
        du[2] = NNSI * S * I - NNIR * I - NNID * I - NNIH * I
        du[3] = NNIR * I + NNHR * H
        du[4] = NNIH * I - NNHR * H - NNHD * H
        du[5] = NNID * I + NNHD * H
    end
    
    return dxdt_pred, p_vec
end

# ========== 3. Training Function with Timing ==========
function train_model(activation_name, activation_fn, u0, t, Infected_Data, Recovered_Data)
    println("\n" * "="^60)
    println("Training with $activation_name activation function")
    println("="^60)
    
    rng = Random.default_rng()
    dxdt_pred, p_init = create_epidemic_model(activation_fn, rng)
    prob_pred = ODEProblem{true}(dxdt_pred, u0, tspan)
    
    function predict_adjoint(θ)
        x = Array(solve(prob_pred, Tsit5(), p=θ, saveat=t,
                       sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
        return x
    end
    
    function loss_adjoint(θ)
        x = predict_adjoint(θ)
        loss = sum(abs2, (Infected_Data .- x[2, :])[2:end])
        loss += sum(abs2, (Recovered_Data .- x[3, :])[2:end])
        return loss
    end
    
    # Training settings
    if activation_name == "ReLU"
        learning_rate = 0.0001
        max_iters = 3000
    elseif activation_name == "Tanh"
        learning_rate = 0.001
        max_iters = 3000
    else  # Sigmoid
        learning_rate = 0.0005
        max_iters = 3000
    end
    
    loss_history = Float64[]
    iter_count = 0
    
    function callback(θ, l)
        iter_count += 1
        push!(loss_history, l)
        if iter_count % 500 == 0
            println("  Iteration $iter_count, Loss: $l")
        end
        return false
    end
    
    # Train and measure time
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p_init)
    
    println("  Starting training...")
    training_time = @elapsed res_adam = Optimization.solve(optprob, 
                        OptimizationOptimisers.Adam(learning_rate), 
                        callback=callback, maxiters=max_iters)
    
    println("  Training completed in $(round(training_time, digits=2)) seconds")
    
    # Final prediction
    final_prediction = predict_adjoint(res_adam.u)
    
    # Calculate errors
    infected_error = mean(abs2, Infected_Data .- final_prediction[2, :])
    recovered_error = mean(abs2, Recovered_Data .- final_prediction[3, :])
    total_error = infected_error + recovered_error
    
    return (loss_history=loss_history, 
            prediction=final_prediction,
            infected_error=infected_error,
            recovered_error=recovered_error,
            total_error=total_error,
            training_time=training_time)
end

# ========== 4. Train All Models ==========
println("\n" * "="^60)
println("STARTING COMPARATIVE TRAINING")
println("="^60)

activations = [
    ("ReLU", relu),
    ("Tanh", tanh),
    ("Sigmoid", sigmoid)
]

results = Dict()
for (name, fn) in activations
    results[name] = train_model(name, fn, u0, t, Infected_Data, Recovered_Data)
end

# ========== 5. Performance Summary ==========
println("\n" * "="^60)
println("PERFORMANCE SUMMARY")
println("="^60)
println("Activation | Infected MSE | Recovered MSE | Total MSE | Time (s)")
println("-"^80)

for (name, res) in results
    println(rpad(name, 10) * " | " *
            rpad(@sprintf("%.6e", res.infected_error), 12) * " | " *
            rpad(@sprintf("%.6e", res.recovered_error), 13) * " | " *
            rpad(@sprintf("%.6e", res.total_error), 10) * " | " *
            @sprintf("%.2f", res.training_time))
end

# Find best performer
best_name = argmin([results[name].total_error for name in keys(results)])
best_error = minimum([results[name].total_error for name in keys(results)])
fastest_name = argmin([results[name].training_time for name in keys(results)])
fastest_time = minimum([results[name].training_time for name in keys(results)])

println("\n🏆 BEST ACCURACY: $best_name (Total MSE: $(round(best_error, sigdigits=4)))")
println("⚡ FASTEST TRAINING: $fastest_name (Time: $(round(fastest_time, digits=2)) seconds)")

# ========== 6. Visualization ==========

# Plot 1: Side-by-side comparison (Infected and Recovered)
p1 = plot(layout=(2,1), size=(1200, 800), titlefontsize=12,
           leftmargin=8Plots.mm, rightmargin=5Plots.mm, bottommargin=8Plots.mm, topmargin=8Plots.mm)

color_dict = Dict("ReLU" => :red, "Tanh" => :blue, "Sigmoid" => :green)

# Infected subplot
plot!(p1[1], title="Infected Population", xlabel="Days", ylabel="Fraction", legend=:topright)
bar!(p1[1], t, Infected_Data, label="True Data", color=:orange, alpha=0.5)

for (name, res) in results
    plot!(p1[1], t, res.prediction[2, :], label="$name (MSE: $(round(res.infected_error, sigdigits=4)))", 
          linewidth=2, color=color_dict[name])
end

# Recovered subplot
plot!(p1[2], title="Recovered Population", xlabel="Days", ylabel="Fraction", legend=:topleft)
bar!(p1[2], t, Recovered_Data, label="True Data", color=:green, alpha=0.5)
for (name, res) in results
    plot!(p1[2], t, res.prediction[3, :], label="$name (MSE: $(round(res.recovered_error, sigdigits=4)))", 
          linewidth=2, color=color_dict[name])
end

savefig("infected_recovered_comparison.png")
display(p1)



# Plot 2: Convergence Speed
p2 = plot(size=(1000, 600), title="Convergence Speed Comparison", 
          xlabel="Iteration", ylabel="Loss (log scale)", yaxis=:log, legend=:topright, 
          leftmargin=8Plots.mm, rightmargin=5Plots.mm, bottommargin=8Plots.mm, topmargin=5Plots.mm)

for (name, res) in results
    plot!(p2, res.loss_history, label="$name (Final: $(round(res.total_error, sigdigits=4)))", 
          linewidth=2, color=color_dict[name])
end

savefig("convergence_comparison.png")
display(p2)

# Plot 3: Individual predictions for each activation
for (name, res) in results
    p_ind = plot(size=(900, 600), title="$name Activation - Predictions vs True",
                 xlabel="Days", ylabel="Population Fraction", legend=:topleft)
    
    # True data
    plot!(p_ind, t, Infected_Data, label="True Infected", linewidth=3, color=:red, linestyle=:dash)
    plot!(p_ind, t, Recovered_Data, label="True Recovered", linewidth=3, color=:blue, linestyle=:dash)
    
    # Predictions
    plot!(p_ind, t, res.prediction[2, :], label="Predicted Infected", linewidth=2, color=:darkred)
    plot!(p_ind, t, res.prediction[3, :], label="Predicted Recovered", linewidth=2, color=:darkblue)
    
    # Add metrics
    annotation_text = "Infected MSE: $(round(res.infected_error, sigdigits=4))\nRecovered MSE: $(round(res.recovered_error, sigdigits=4))\nTime: $(round(res.training_time, digits=2))s"
    annotate!(p_ind, maximum(t)*0.7, maximum(Infected_Data)*0.85, text(annotation_text, 11, :black))
    
    savefig("$(lowercase(name))_predictions.png")
    display(p_ind)
end

# ========== 7. Final Summary ==========
println("\n" * "="^60)
println("FINAL SUMMARY")
println("="^60)

# Create a summary DataFrame-like output
println("\nMetric                | ReLU          | Tanh          | Sigmoid")
println("-"^70)

# MSE values
println("Infected MSE         | $(@sprintf("%.6e", results["ReLU"].infected_error)) | $(@sprintf("%.6e", results["Tanh"].infected_error)) | $(@sprintf("%.6e", results["Sigmoid"].infected_error))")
println("Recovered MSE        | $(@sprintf("%.6e", results["ReLU"].recovered_error)) | $(@sprintf("%.6e", results["Tanh"].recovered_error)) | $(@sprintf("%.6e", results["Sigmoid"].recovered_error))")
println("Total MSE            | $(@sprintf("%.6e", results["ReLU"].total_error)) | $(@sprintf("%.6e", results["Tanh"].total_error)) | $(@sprintf("%.6e", results["Sigmoid"].total_error))")
println("Training Time (s)    | $(round(results["ReLU"].training_time, digits=2))          | $(round(results["Tanh"].training_time, digits=2))          | $(round(results["Sigmoid"].training_time, digits=2))")

println("\n" * "="^60)
println("COMPARISON COMPLETE!")
println("="^60)
println("\nGenerated plots:")
println("  1. infected_recovered_comparison.png - Side-by-side comparison")
println("  2. convergence_comparison.png - Training convergence")
println("  3. relu_predictions.png, tanh_predictions.png, sigmoid_predictions.png - Individual plots")

# Save results to a JLD2 file for future reference
save("comparison_results.jld2", "results", results, "t", t, 
     "Infected_Data", Infected_Data, "Recovered_Data", Recovered_Data)

println("\nResults saved to comparison_results.jld2")

# For reloading results in the future, you can use:
#== results = load("comparison_results.jld2", "results")
 t = load("comparison_results.jld2", "t")
 Infected_Data = load("comparison_results.jld2", "Infected_Data")
 Recovered_Data = load("comparison_results.jld2", "Recovered_Data")

println("\nTo reload results, use:")
println("  results = load(\"comparison_results.jld2\", \"results\")")
println("  t = load(\"comparison_results.jld2\", \"t\")")
println("  Infected_Data = load(\"comparison_results.jld2\", \"Infected_Data\")")
println("  Recovered_Data = load(\"comparison_results.jld2\", \"Recovered_Data\")")

==#
