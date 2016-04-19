using DataFrames
using HypothesisTests
using Lazy
using Logging
using MLBase
using Optim

include("DataInfo.jl")
include("helpers.jl")
include("svr_base.jl")


function calc_scores_gen(
    ixs::AbstractVector{Int64},
    o::Outcome,
    dataset::DataSet;
    n_iters::Int64 = 10,
    seed::Nullable{Int}=Nullable(1234))

  X::Matrix{Float64}, y::Vector{Float64} = begin
    Xy::XY = get_Xy_mat(o, diff_wpm,
                        dataset=dataset,
                        region=left_select,
                        subject_group=all_subjects)
    Xy[1][ixs, :], Xy[2][ixs]
  end

  num_samples = length(y)
  cvg::CrossValGenerator = get_cvg(RandomSub, n_iters, num_samples)

  debug("num_samples: $num_samples")

  svr::PyObject = LinearSVR()

  pred(inds::Vector{Int64}) = r2_score(y[inds], svr[:predict](X[inds, :]))

  test(_, inds::Vector{Int64}) = pred(inds)

  function calc_scores(C::Float64)
    @set_seed

    svr[:C] = C

    train_scores = zeros(Float64, n_iters)
    fit_call_count::Int64 = 0
    fit(inds::Vector{Int64}) = begin
      fit_call_count += 1
      svr[:fit](X[inds, :], y[inds])
      train_scores[fit_call_count] = pred(inds)
    end

    test_scores = cross_validate(fit, test, num_samples, cvg)

    debug("C: $C")
    debug("mean test scores: $(mean(test_scores))")
    debug("mean train scores: $(mean(train_scores))")

    test_scores, train_scores
  end
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


function simple_newton(fn::Function, y::Float64,
                      min_x::Float64, max_x::Float64;
                       n_iters::Int64 = 100,
                       min_delta_ratio::Float64 = .1,
                       _recursion_state::NewtionRecusionState = NewtionRecusionState())

  @assert max_x > min_x

  mid_x::Float64 = (max_x + min_x)/2
  y_guess::Float64 = fn(mid_x)

  curr_diff::Float64 = abs(y - y_guess)

  _recursion_state.current_iter += 1

  if curr_diff < _recursion_state.best_diff
    _recursion_state.best_diff = curr_diff
    _recursion_state.best_x = mid_x
  end

  if curr_diff < min_delta_ratio || _recursion_state.current_iter >= n_iters
    return _recursion_state.best_x
  end

  max_x::Float64, min_x::Float64 = y_guess > y ? (mid_x, min_x) : (max_x, mid_x)

  simple_newton(fn, y, min_x, max_x, min_delta_ratio=min_delta_ratio,
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

function find_optimum_C(ixs::AbstractVector{Int64},
                        o::Outcome, d::DataSet;
                        c_range::Nullable{CRange} = Nullable{CRange}())
  calc_scores_g(n_iters::Int64) = calc_scores_gen(ixs, o, d, n_iters=n_iters)

  min_C::Float64, max_C::Float64 = begin
    if !isnull(c_range)
      c_range_val::CRange = get(c_range)
      c_range_val.min_x, c_range_val.max_x
    else

      n_iters = 10
      get_train_scores_mn(C::Float64) = mean(calc_scores_g(n_iters)(C)[2])
      min_score::Float64 = .2
      max_score::Float64 = .8

      ret_max::Float64 = simple_newton(get_train_scores_mn, max_score,
                                       5e-4, 5000.)

      debug("max C: $ret_max")

      ret_min::Float64 = simple_newton(get_train_scores_mn, min_score,
                                       5e-4, ret_max)

      debug("min C: $ret_min")

      ret_min, ret_max
    end
  end

  debug("will now find best test score")

  get_test_scores(C::Float64) = calc_scores_g(50)(C)[1]
  get_test_scores_t(C::Float64) = -1. * OneSampleTTest(get_test_scores(C)).t
  C::Float64 = optimize(get_test_scores_t, min_C, max_C, rel_tol=.1).minimum

  (C, min_C, max_C)
end


function find_optimum_C(ixs::AbstractVector{Int64},
                        di::DataInfo;
                        c_range::Nullable{CRange} = Nullable{CRange}())
  find_optimum_C(ixs, di.outcome, di.dataset, c_range=c_range)
end


typealias DataSetMap Dict{DataSet, Float64}

function ensemble(outcome::Outcome, get_C::Function,
                  region::Region=left_select;
                  weights::DataSetMap = Dict(conn => .5, lesion => .5),
                  n_schemes::Int64=1000,
                  seed::Nullable{Int}=Nullable(1234))

  @set_seed

  ds::AbstractVector{DataSet} = collect(keys(weights))

  all_ds(f::Function, T::Type) = Dict{DataSet, T}(
    [d => f(d)::T for d in ds])

  svrs::Dict{DataSet, PyObject} = all_ds(d -> LinearSVR(), PyObject)

  dis::Dict{DataSet, DataInfo} = all_ds(DataInfo) do d::DataSet
    DataInfo(outcome, diff_wpm, all_subjects, region, d)
  end

  typealias Ids AbstractVector{UTF8String}
  ids::Dict{DataSet, Ids} = all_ds(Ids) do d
    get_full(dis[d])[:id]
  end
  common_ids::Vector{UTF8String} = intersect(values(ids)...)

  common_id_ixs(d::DataSet) = map(common_ids) do id
    ret::Vector{Int64} = find(ids[d] .== id)
    @assert length(ret) == 1
    ret[1]
  end

  XYs::Dict{DataSet, XY} = all_ds(XY) do d::DataSet
    di::DataInfo = dis[d]
    (X::Matrix{Float64}, y::Vector{Float64}) = get_Xy_mat(di)
    ixs::Vector{Int64} = common_id_ixs(d)
    X[ixs, :], y[ixs]
  end

  main_y::Vector{Float64} = collect(values(XYs))[1][2]
  all_ys_same::Bool = begin
    is_same_as_main(acc::Bool, xy::XY) = acc && xy[2] == main_y
    reduce(is_same_as_main, true, values(XYs))
  end
  @assert all_ys_same

  scheme_ix::Int64 = 0
  function fit(inds::Vector{Int64})
    scheme_ix += 1
    all_ds(PyObject) do d::DataSet
      info("about to fit")

      C::Float64 = get_C(dis[d], inds, scheme_ix)

      svr::PyObject = svrs[d]
      svr[:C] = C
      info("fit C: $C")
      X::Matrix{Float64} = XYs[d][1]
      svr[:fit](X[inds, :], main_y[inds])
    end
  end

  test(svrs::Dict{DataSet, PyObject}, inds::Vector{Int64}) = begin
    predictions::Matrix{Float64} = begin
      ret = zeros(Float64, length(inds), 2)
      for (ix::Int64, (d::DataSet, svr::PyObject)) in enumerate(svrs)
        X::Matrix{Float64} = XYs[d][1]
        ret[:, ix] = svr[:predict](X[inds, :]) .* weights[d]
      end
      ret
    end

    ensemble_predictions::Vector{Float64} = sum(predictions, 2)[:]
    @assert length(ensemble_predictions) == length(inds)

    r2_score(main_y[inds], ensemble_predictions)
  end

  num_samples::Int64 = length(common_ids)

  cvg::CrossValGenerator = get_cvg(RandomSub, n_schemes, num_samples)
  cross_validate(fit, test, num_samples, cvg)
end


type State
  set_ixs::Vector{Bool}
  cs::Vector{Float64}
  c_range::Nullable{CRange}
end

function State(n_schemes::Int64)
  State(zeros(Bool, n_schemes),
        zeros(Float64, n_schemes),
        Nullable{CRange}())
end

function run(n_schemes::Int64=1000)

  state_map::Dict{DataInfo, State} = Dict{DataInfo, State}()

  get_C::Function = begin
    typealias CRangeMap Dict{DataInfo, Nullable{CRange}}
    prev_range_map::CRangeMap = CRangeMap()

    function update_state!(di::DataInfo, scheme_ix::Int64,
                          C::Float64, min_C::Float64, max_C::Float64)
      state::State = get(state_map, di, State(n_schemes))
      state.set_ixs[scheme_ix] = true

      prev_range::Nullable{CRange} = state.c_range
      if isnull(prev_range)
        state.c_range = Nullable(CRange(min_C/1.5, max_C*1.5))
      end

      state.cs[scheme_ix] = C

      state_map[di] = state

    end

    fn(di::DataInfo, ixs::AbstractVector{Int64}, scheme_ix::Int64) = begin
      state::State = get(state_map, di, State(n_schemes))
      if state.set_ixs[scheme_ix]
        return state.cs[scheme_ix]
      end

      C::Float64, min_C::Float64, max_C::Float64 = find_optimum_C(
        ixs, di, c_range=state.c_range)

      update_state!(di, scheme_ix, C, min_C, max_C)

      C
    end
  end

  pred(weights::Dict{DataSet, Float64}) = ensemble(adw, get_C,
                                                   weights=weights, n_schemes=n_schemes)

  println("predicting ensemble")
  pred_55 = pred(Dict(conn => .5, lesion => .5))

  println("predicting lesion")
  pred_lesion = pred(Dict(lesion => 1.))

  println("predicting connection")
  pred_conn = pred(Dict(conn => 1.))

  ret = Dict()
  ret[:preds] = Dict(:55 => pred_55, :conn => pred_conn, :lesion => pred_lesion)
  ret[:state] = state_map
  ret
end
