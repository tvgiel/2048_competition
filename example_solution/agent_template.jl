using Random

include("../game2048.jl")

struct AIModel
end

function act(agent::AIModel, game_state, valid_actions)
    return rand(valid_actions)
end

# Some models do not need training. Return the initialized model in that case.
function train_and_instantiate()
    return AIModel()
end

function play_game(agent::AIModel=train_and_instantiate())
    env = Game2048Env()
    state = reset!(env)
    info = Dict("score" => env.score, "highest_tile" => maximum(env.board))

    while !env.done
        valid_actions = get_available_actions(env)
        if isempty(valid_actions)
            break
        end

        action = act(agent, state, valid_actions)
        state, reward, done, info = step!(env, action)
    end

    return info["score"], info["highest_tile"]
end
