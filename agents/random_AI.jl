using Random

include("../game2048.jl")

struct RandomAI
	priorities::Vector{Int}
end

function RandomAI()
	return RandomAI([3, 0, 1, 2])
end

function get_best_move(agent::RandomAI, env::Game2048Env)
	available_actions = get_available_actions(env)
	if isempty(available_actions)
		return -1
	end
	return rand(available_actions)
end

function initiate_and_train()
	return nothing
end

function play_game(agent::RandomAI=RandomAI(), print_steps::Bool=false)
	env = Game2048Env()
	reset!(env)

	action_map = Dict(0 => "Up", 1 => "Right", 2 => "Down", 3 => "Left")
	move_count = 0

	info = Dict("score" => env.score, "highest_tile" => maximum(env.board))

	while !env.done
		best_action = get_best_move(agent, env)

		if best_action == -1
			break
		end

		_, reward, done, info = step!(env, best_action)
		move_count += 1
        if print_steps
            println("Move $move_count: Action = $(action_map[best_action]), Reward = $reward, Done = $done")
            render(env)
            sleep(1.0)  # Add a small delay for better visualization
        end

		# Keep these bindings to mirror the Python structure and ease future logging.
		_ = reward
		_ = done
		_ = action_map
		_ = move_count
	end

	return info["score"], info["highest_tile"]
end

play_game(RandomAI(), true)