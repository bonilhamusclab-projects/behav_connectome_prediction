using DataFrames
using HypothesisTests
using Lazy
using Logging
using MLBase
using Optim

include("DataInfo.jl")
include("helpers.jl")
include("svrBase.jl")


function calcScoresGen(
    X::Matrix{Float64}, y::Vector{Float64};
    n_iters::Int64 = 10,
    seed::Nullable{Int}=Nullable(1234))

  n_samples = length(y)
  cvg::CrossValGenerator = getCvg(RandomSub, n_iters, n_samples)

  svr::PyObject = LinearSVR()

  pred(inds::Vector{Int64}) = r2Score(y[inds], svr[:predict](X[inds, :]))

  test(_, inds::Vector{Int64}) = pred(inds)

  function calcScores(C::Float64)

    svr[:C] = C

    train_scores = zeros(Float64, n_iters)
    fit_call_count::Int64 = 0
    fit(inds::Vector{Int64}) = begin
      fit_call_count += 1
      svr[:fit](X[inds, :], y[inds])
      train_scores[fit_call_count] = pred(inds)
    end

    test_scores = cross_validate(fit, test, n_samples, cvg)

    debug(@sprintf "C: %3.2e, test mean: %3.2e test T: %3.2e, train mean: %3.2e" C mean(test_scores) OneSampleTTest(test_scores).t mean(train_scores))

    test_scores, train_scores
  end
end


function calcScoresGen(ixs::AbstractVector{Int64},
  o::Outcome,
  dataset::DataSet;
  n_iters::Int64 = 10,
  seed::Nullable{Int}=Nullable(1234))

  X::Matrix{Float64}, y::Vector{Float64} = begin
    Xy::XY = getXyMat(o, diff_wpm,
                      dataset=dataset,
                      region=left_select2,
                      subject_group=all_subjects)
    Xy[1][ixs, :], Xy[2][ixs]
  end

  calcScoresGen(X, y, n_iters=n_iters, seed=seed)
end


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

immutable CRange
  min_x::Float64
  max_x::Float64

  CRange(min_x::Float64, max_x::Float64) = begin
    @assert min_x <= max_x
    new(min_x, max_x)
  end
end


function findOptimumC(X::Matrix{Float64}, y::Vector{Float64};
                      c_range::Nullable{CRange} = Nullable{CRange}(),
                      score_train_scores::Function = mean,
                      score_test_scores::Function = arr::AbstractVector{Float64} -> OneSampleTTest(arr).t
                      )
  calcScoresG(n_iters::Int64) = calcScoresGen(X, y, n_iters=n_iters)

  min_C::Float64, max_C::Float64 = begin
    if !isnull(c_range)
      c_range_val::CRange = get(c_range)
      c_range_val.min_x, c_range_val.max_x
    else

      n_iters = 10
      scoreTrainScores(C::Float64) = score_train_scores(calcScoresG(n_iters)(C)[2])
      min_score::Float64 = .2
      max_score::Float64 = .8

      ret_max::Float64 = simpleNewton(scoreTrainScores, max_score,
                                       5e-4, 5000.)


      ret_min::Float64 = simpleNewton(scoreTrainScores, min_score,
                                       5e-4, ret_max)


      ret_min, ret_max
    end
  end


  getTestScores(C::Float64) = calcScoresG(50)(C)[1]
  getTestScoresT(C::Float64) = -1. * score_test_scores(getTestScores(C))
  C::Float64 = optimize(getTestScoresT, min_C, max_C, rel_tol=.1).minimum

  info(@sprintf "best C: %3.2e, with t: %3.2e, for dataset %s" C -1 * getTestScoresT(C) d)

  (C, min_C, max_C)
end


function findOptimumC(X, y;
                      c_range::Nullable{CRange} = Nullable{CRange}(),
                      score_train_scores::Function = mean,
                      score_test_scores::Function =
                        arr::AbstractVector{Float64} -> OneSampleTTest(arr).t)
  findOptimumC(X::Matrix{Float64}, y::Vector{Float64},
                 c_range=c_range, score_train_scores=score_train_scores,
                 score_test_scores=score_test_scores)
end


typealias DataSetMap Dict{DataSet, Float64}

function ensemble(outcome::Outcome,
                  get_C::Function,
                  region::Region=left_select2;
                  weights::DataSetMap = Dict(conn => .5, lesion => .5),
                  n_repetitions::Int64=1000,
                  seed::Nullable{Int}=Nullable(1234),
                  is_perm::Bool=false)

  @set_seed

  ds::AbstractVector{DataSet} = collect(keys(weights))

  allDs(f::Function, T::Type) = Dict{DataSet, T}(
    [d => f(d)::T for d in ds])

  svrs::Dict{DataSet, PyObject} = allDs(d -> LinearSVR(), PyObject)

  dis::Dict{DataSet, DataInfo} = allDs(DataInfo) do d::DataSet
    DataInfo(outcome, diff_wpm, all_subjects, region, d)
  end

  XYs::Dict{DataSet, XY} = allDs(XY) do d::DataSet
    di::DataInfo = dis[d]
    (X::Matrix{Float64}, y::Vector{Float64}) = getXyMat(di)
    X, y
  end


  common_ids::Ids = begin
    ids::Dict{DataSet, Ids} = allDs(Ids) do d
      getFull(dis[d])[:id]
    end
    intersect(values(ids)...)
  end

  getRepetitionIds = getRepetitionSamplesGen(common_ids, n_repetitions, is_perm)

  function fit(repetition_ix::Int64)

    (X_train_ids::Ids, y_train_ids::Ids) = getRepetitionIds(repetition_ix)[1]

    allDs(PyObject) do d::DataSet
      di::DataInfo = dis[d]
      X_ixs::Vector{Int64} = getIdIxs(di, X_train_ids)
      y_ixs::Vector{Int64} = getIdIxs(di, y_train_ids)

      X::Matrix{Float64} = XYs[d][1][X_ixs, :]
      y::Vector{Float64} = XYs[d][2][y_ixs]

      C::Float64 = get_C(X, y, di, repetition_ix)

      svr::PyObject = svrs[d]
      svr[:C] = C
      svr[:fit](X, y)
    end
  end

  function test(repetition_ix)

    (test_x_inds::Ids, test_y_inds::Ids) = getRepetitionIds(repetition_ix)[2]
    num_samples = length(test_x_inds)
    @assert num_samples == length(test_y_inds)

    getY(d::DataSet) = XYs[d][2][getIdIxs(dis[d], test_y_inds)]
    getX(d::DataSet) = XYs[d][1][getIdIxs(dis[d], test_x_inds), :]

    predictions::Matrix{Float64} = begin
      ret = zeros(Float64, num_samples, length(svrs))
      for (ix::Int64, (d::DataSet, svr::PyObject)) in enumerate(svrs)
        X::Matrix{Float64} = getX(d)
        ret[:, ix] = svr[:predict](X) .* weights[d]
      end
      ret
    end
    ensemble_predictions::Vector{Float64} = sum(predictions, 2)[:]

    y::Vector{Float64} = begin
      first_y::Vector{Float64} = getY(ds[1])
      @assert reduce(true, ds) do acc::Bool, d::DataSet
        acc & (getY(d) == first_y)
      end

      first_y
    end

    @assert length(ensemble_predictions) == length(y)

    r2Score(y, ensemble_predictions)
  end

  typealias IndsPair Tuple{Vector{Int64}}
  scores::Vector{Float64} = map(1:n_repetitions) do r::Int64
    fit(r)
    test(r)
  end

  scores, getRepetitionIds
end


type State
  set_ixs::Vector{Bool}
  cs::Vector{Float64}
  c_range::Nullable{CRange}
  getids::Function
end

function State(n_repetitions::Int64)
  State(zeros(Bool, n_repetitions),
        zeros(Float64, n_repetitions),
        Nullable{CRange}(),
        i::Int64 -> error("not set yet")
        )
end

function run(n_repetitions::Int64=1000, is_perm=false)

  state_map::Dict{DataInfo, State} = Dict{DataInfo, State}()

  get_C::Function = begin
    typealias CRangeMap Dict{DataInfo, Nullable{CRange}}
    prev_range_map::CRangeMap = CRangeMap()

    function updateState!(di::DataInfo, repetition_ix::Int64,
                          C::Float64, min_C::Float64, max_C::Float64)
      state::State = get(state_map, di, State(n_repetitions))
      state.set_ixs[repetition_ix] = true

      prev_range::Nullable{CRange} = state.c_range
      if isnull(prev_range)
        state.c_range = Nullable(CRange(min_C/1.5, max_C*1.5))
      end

      state.cs[repetition_ix] = C

      state_map[di] = state

    end

    fn(X::Matrix{Float64}, y::Vector{Float64}, di, repetition_ix::Int64) = begin
      state::State = get(state_map, di, State(n_repetitions))
      if state.set_ixs[repetition_ix]
        return state.cs[repetition_ix]
      end

      debug("datainfo: $di")
      C::Float64, min_C::Float64, max_C::Float64 = findOptimumC(
        X, y, c_range=state.c_range)

      updateState!(di, repetition_ix, C, min_C, max_C)

      C
    end
  end

  pred(weights::Dict{DataSet, Float64}) = ensemble(adw, get_C,
                                                   weights=weights,
                                                   n_repetitions=n_repetitions,
                                                   is_perm=is_perm)

  println("predicting ensemble")
  pred_55, get_ids_55 = pred(Dict(conn => .5, lesion => .5))

  println("predicting lesion")
  pred_lesion, get_ids_lesion = pred(Dict(lesion => 1.))

  println("predicting connection")
  pred_conn, get_ids_conn = pred(Dict(conn => 1.))

  ret = Dict()
  ret[:preds] = Dict(:55 => pred_55, :conn => pred_conn, :lesion => pred_lesion)
  ret[:state] = state_map

  mkDi(ds::DataSet) = DataInfo(adw, diff_wpm, all_subjects, left_select2, ds)
  ret[:state][mkDi(conn)].getids = get_ids_conn
  ret[:state][mkDi(lesion)].getids = get_ids_lesion

  ret
end
