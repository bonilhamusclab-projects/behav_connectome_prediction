using DataFrames
using HypothesisTests
using Lazy
using Logging
using MLBase
using Optim

include("DataInfo.jl")
include("helpers.jl")
include("optimizeHelpers.jl")
include("svrBase.jl")


function calcScoresGen(
    X::Matrix{Float64}, y::Vector{Float64};
    n_iters::Int64 = 10,
    seed::Nullable{Int}=Nullable(1234))

  n_samples = length(y)
  cvg::CrossValGenerator = getCvg(RandomSub, n_iters, n_samples)

  pipe::Pipeline = svrPipelineGen(X, y)

  test(_, ixs::Vector{Int64}) = pipeTest(pipe, ixs)

  function calcScores(C::Float64)
    paramState(pipe, :svr)[:C] = C

    train_scores = zeros(Float64, n_iters)
    fit_call_count::Int64 = 0
    fit(ixs::IXs) = begin
      fit_call_count += 1
      pipeFit!(pipe, ixs)

      train_scores[fit_call_count] = pipeTest(pipe, ixs)
    end

    test_scores = cross_validate(fit, test, n_samples, cvg)

    debug(@sprintf "C: %3.2e, test mean: %3.2e test T: %3.2e, train mean: %3.2e" C mean(test_scores) OneSampleTTest(test_scores).t mean(train_scores))

    test_scores, train_scores
  end
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
  calcScoresGenIters(n_iters) = calcScoresGen(X, y, n_iters=n_iters)

  min_C::Float64, max_C::Float64 = begin
    if !isnull(c_range)
      c_range_val::CRange = get(c_range)
      c_range_val.min_x, c_range_val.max_x
    else

      calc10Scores = calcScoresGenIters(10)
      scoreTrainScores(C::Float64) = calc10Scores(C)[2] |> score_train_scores

      min_score::Float64 = .2
      max_score::Float64 = .8

      ret_max::Float64 = simpleNewton(scoreTrainScores, max_score,
                                       5e-4, 5000.)

      ret_min::Float64 = simpleNewton(scoreTrainScores, min_score,
                                       5e-4, ret_max)


      ret_min, ret_max
    end
  end


  calc50Scores = calcScoresGenIters(50)
  getTestScores(C::Float64) = calc50Scores(C)[1]
  getTestScoresT(C::Float64) = -1. * score_test_scores(getTestScores(C))
  C::Float64 = optimize(getTestScoresT, min_C, max_C, rel_tol=.1).minimum

  info(@sprintf "best C: %3.2e, with t: %3.2e, for dataset %s" C -1 * getTestScoresT(C) d)

  (C, min_C, max_C)
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

  ds::AbstractVector{DataSet} = weights |> keys |> collect

  allDs(f::Function, T::Type) = Dict{DataSet, T}(
    [d => f(d)::T for d in ds])

  dis::Dict{DataSet, DataInfo} = allDs(DataInfo) do d::DataSet
    DataInfo(outcome, diff_wpm, all_subjects, region, d)
  end

  common_ids::Ids = dis |> values |> collect |> getCommonIds

  XYs::Dict{DataSet, XY} = allDs(XY) do d
    X, y =  getXyMat(dis[d], common_ids)
    X, y
  end

  pipes::Dict{DataSet, Pipeline} = allDs(Pipeline) do d
    X, y = XYs[d]
    svrPipelineGen(X, y)
  end

  getRepetitionIxs = getRepetitionSamplesGen(length(common_ids), n_repetitions, is_perm)

  function fit(repetition_ix::Int64)

    (X_train_ixs::IXs, y_train_ixs::IXs) = getRepetitionIxs(repetition_ix)[1]

    for (d, pipe) in pipes
      X, y = XYs[d]
      C = get_C(X[X_train_ixs, :], y[y_train_ixs], dis[d], repetition_ix)

      paramState(pipe, :svr)[:C] = C
      pipeFit!(pipe, X_train_ixs, y_train_ixs)
    end

  end

  function test(repetition_ix)
    (test_x_ixs::IXs, test_y_ixs::IXs) = getRepetitionIxs(repetition_ix)[2]
    num_samples = length(test_x_ixs)
    @assert num_samples == length(test_y_ixs)

    predictions::Matrix{Float64} = begin
      ret = zeros(Float64, num_samples, length(pipes))
      for (ix::Int64, (d::DataSet, pipe::Pipeline)) in enumerate(pipes)
        ret[:, ix] = pipePredict(pipe, test_x_ixs, test_y_ixs) .* weights[d]
      end
      ret
    end

    ensemble_predictions::Vector{Float64} = sum(predictions, 2)[:]

    y::Vector{Float64} = begin
      getY(Xy::XY) = Xy[2][test_y_ixs]

      first_y::Vector{Float64} = XYs |> values |> first |> getY
      @assert reduce(true, values(XYs)) do acc::Bool, Xy::XY
        acc & (getY(Xy) == first_y)
      end

      first_y
    end

    @assert length(ensemble_predictions) == length(y)

    r2Score(y, ensemble_predictions)
  end

  scores::Vector{Float64} = map(1:n_repetitions) do r::Int64
    fit(r)
    test(r)
  end

  scores, common_ids, getRepetitionIxs
end


type State
  set_ixs::Vector{Bool}
  cs::Vector{Float64}
  c_range::Nullable{CRange}
  getrepixs::Function
  ids::Ids
end

function State(n_repetitions::Int64)
  State(zeros(Bool, n_repetitions),
        zeros(Float64, n_repetitions),
        Nullable{CRange}(),
        i::Int64 -> error("not set yet"),
        ASCIIString[]
        )
end


function findOptimumCandUpdateStateGen(state_map::Dict{DataInfo, State},
  n_repetitions)

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


function run(n_repetitions::Int64=1000, is_perm=false)

  state_map::Dict{DataInfo, State} = Dict{DataInfo, State}()

  get_C!::Function = findOptimumCandUpdateStateGen(state_map, n_repetitions)

  pred(weights::Dict{DataSet, Float64}) = ensemble(adw, get_C!,
                                                   weights=weights,
                                                   n_repetitions=n_repetitions,
                                                   is_perm=is_perm)

  println("predicting ensemble")
  pred_55, ids_55, get_ixs_55 = pred(Dict(conn => .5, lesion => .5))

  println("predicting lesion")
  pred_lesion, ids_lesion, get_ixs_lesion = pred(Dict(lesion => 1.))

  println("predicting connection")
  pred_conn, ids_conn, get_ixs_conn = pred(Dict(conn => 1.))

  ret = Dict()
  ret[:preds] = Dict(:55 => pred_55, :conn => pred_conn, :lesion => pred_lesion)
  ret[:state] = state_map

  mkDi(ds::DataSet) = DataInfo(adw, diff_wpm, all_subjects, left_select2, ds)

  ret[:state][mkDi(conn)].getrepixs = get_ixs_conn
  ret[:state][mkDi(conn)].ids = ids_conn

  ret[:state][mkDi(lesion)].getrepixs = get_ixs_lesion
  ret[:state][mkDi(lesion)].ids = ids_lesion

  ret
end
