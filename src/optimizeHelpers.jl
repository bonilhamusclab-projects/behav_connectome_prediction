using Logging


type NewtionRecusionState
  best_x::Float64
  best_diff::Float64
  current_iter::Int64
end

function NewtionRecusionState(;best_x::Float64 = Inf,
                              best_diff::Float64=Inf,
                              current_iter::Int64 = 1)
  NewtionRecusionState(best_x, best_diff, current_iter)
end


function simpleNewton(fn::Function, y::Float64,
                      min_x::Float64, max_x::Float64;
                       n_iters::Int64 = 100,
                       min_delta_ratio::Float64 = .05,
                       _recursion_state::NewtionRecusionState = NewtionRecusionState())

  @assert max_x > min_x

  mid_x::Float64 = (max_x + min_x)/2
  y_guess::Float64 = fn(mid_x)

  debug(@sprintf "Newton iter: %d, y_guess: %3.2f, y: %3.2f" _recursion_state.current_iter y_guess y)
  debug(@sprintf "Newton mid_x: %3.2e" mid_x)

  curr_diff::Float64 = abs(y - y_guess)

  _recursion_state.current_iter += 1

  if curr_diff < _recursion_state.best_diff
    _recursion_state.best_diff = curr_diff
    _recursion_state.best_x = mid_x
  end

  max_x::Float64, min_x::Float64 = y_guess > y ? (mid_x, min_x) : (max_x, mid_x)

  newton_done = curr_diff < min_delta_ratio ||
    _recursion_state.current_iter >= n_iters ||
    max_x <= min_x

  if newton_done
    debug("newton done")
    return _recursion_state.best_x
  end

  simpleNewton(fn, y, min_x, max_x, min_delta_ratio=min_delta_ratio,
                _recursion_state=_recursion_state)
end
