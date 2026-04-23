using Random

mutable struct Game2048Env
	size::Int
	board::Matrix{Int}
	score::Int
	done::Bool
end

function Game2048Env(size::Int=4)
	board = zeros(Int, size, size)
	return Game2048Env(size, board, 0, false)
end

function reset!(env::Game2048Env)
	env.board .= 0
	env.score = 0
	env.done = false
	_spawn_tile!(env)
	_spawn_tile!(env)
	return get_state(env)
end

function _rot90k(board::Matrix{Int}, k::Int)
	kk = mod(k, 4)
	if kk == 0
		return copy(board)
	elseif kk == 1
		return rotl90(board)
	elseif kk == 2
		return rot180(board)
	else
		return rotr90(board)
	end
end

function step!(env::Game2048Env, action::Int)
	if env.done
		return get_state(env), 0, env.done, Dict("error" => "Game is already over")
	end

	original_board = copy(env.board)
	reward = 0

	rotation_k = mod(action + 1, 4)
	env.board = _rot90k(env.board, rotation_k)

	for i in 1:env.size
		new_row, row_reward = _slide_and_merge(env, vec(env.board[i, :]))
		env.board[i, :] = new_row
		reward += row_reward
	end

	env.board = _rot90k(env.board, -rotation_k)

	valid_move = original_board != env.board
	if valid_move
		env.score += reward
		_spawn_tile!(env)
	else
		reward = -1
	end

	env.done = _check_game_over(env)

	info = Dict(
		"score" => env.score,
		"highest_tile" => maximum(env.board),
		"valid_move" => valid_move,
	)

	return get_state(env), reward, env.done, info
end

function get_available_actions(env::Game2048Env)
	actions = Int[]
	for action in 0:3
		rotation_k = mod(action + 1, 4)
		test_board = _rot90k(env.board, rotation_k)
		changed = false

		for i in 1:env.size
			new_row, _ = _slide_and_merge(env, vec(test_board[i, :]))
			if vec(test_board[i, :]) != new_row
				changed = true
				break
			end
		end

		if changed
			push!(actions, action)
		end
	end
	return actions
end

function get_state(env::Game2048Env)
	return copy(env.board)
end

function _slide_and_merge(env::Game2048Env, row::Vector{Int})
	non_zero = [x for x in row if x != 0]
	reward = 0

	merged_row = Int[]
	i = 1
	while i <= length(non_zero)
		if i < length(non_zero) && non_zero[i] == non_zero[i + 1]
			merged_val = non_zero[i] * 2
			push!(merged_row, merged_val)
			reward += merged_val
			i += 2
		else
			push!(merged_row, non_zero[i])
			i += 1
		end
	end

	append!(merged_row, zeros(Int, env.size - length(merged_row)))
	return merged_row, reward
end

function _spawn_tile!(env::Game2048Env)
	empty_cells = findall(==(0), env.board)
	if !isempty(empty_cells)
		idx = rand(empty_cells)
		env.board[idx] = rand() < 0.1 ? 4 : 2
	end
end

function _check_game_over(env::Game2048Env)
	if any(==(0), env.board)
		return false
	end

	for i in 1:env.size
		for j in 1:(env.size - 1)
			if env.board[i, j] == env.board[i, j + 1]
				return false
			end
		end
	end

	for i in 1:(env.size - 1)
		for j in 1:env.size
			if env.board[i, j] == env.board[i + 1, j]
				return false
			end
		end
	end

	return true
end

function render(env::Game2048Env)
	println(repeat("-", 25))
	for row in eachrow(env.board)
		row_str = join([num != 0 ? lpad(string(num), 5) : "     " for num in row], "|")
		println("|" * row_str * "|")
	end
	println(repeat("-", 25))
	println("Score: $(env.score) | Max Tile: $(maximum(env.board))\n")
end

