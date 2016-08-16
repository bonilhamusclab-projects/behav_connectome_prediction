using DataFrames
using Lazy
using Memoize
using MLBase
using PyCall

include("DataInfo.jl")
include("helpers.jl")
include("mlHelpers.jl")

@pyimport sklearn.svm as svm
LinearSVR = svm.LinearSVR
LinearSVC = svm.LinearSVC
SVC = svm.SVC


@memoize function getXyMat(o::Outcome,
                    target::Target;
                    dataset::DataSet=conn,
                    region::Region=full_brain,
                    subject_group::SubjectGroup=all_subjects)
  getXyMat(DataInfo(o, target, subject_group, region, dataset))
end


function learningCurve(svr::PyObject,
                        lc_cvg_gen::Function,
                        o::Outcome,
                        dataset::DataSet,
                        region::Region,
                        subject_group::SubjectGroup,
                        train_ratios::AbstractVector{Float64};
                        t::Target=diff_wpm,
                        score_fn::Function=r2Score)

  X, y = getXyMat(o, t, dataset=dataset, region=region, subject_group=subject_group)
  num_samples::Int64 = length(y)

  train_sizes::Vector{Int64} = ratiosToCounts(train_ratios, num_samples)
  train_sizes = train_sizes[train_sizes .> 2]

  ret::DataFrame = begin
    mkZeros = () -> zeros(Float64, length(train_sizes))

    DataFrame(train_mn=mkZeros(),
              train_std=mkZeros(),
              test_mn=mkZeros(),
              test_std=mkZeros(),
              train_size=train_sizes)
  end

  test(c::PyObject, inds::Vector{Int64}) =
    score_fn(y[inds], c[:predict](X[inds, :]))

  for (ix, t) in enumerate(train_sizes)
    cvg::CrossValGenerator = lc_cvg_gen(t)

    train_scores = Float64[]

    fit(inds::Vector{Int64}) = begin
      svr[:fit](X[inds, :], y[inds])
      train_score = test(svr, inds)
      train_scores = [train_scores; train_score]
      svr
    end

    test_scores = cross_validate(fit, test, t, cvg)

    ret[ix, :train_mn] = mean(train_scores)
    ret[ix, :train_std] = std(train_scores)
    ret[ix, :test_mn] = mean(test_scores)
    ret[ix, :test_std] = std(test_scores)
  end

  ret
end


function getCvgGen(cvgT::Type, n_folds::Int64)
  @assert cvgT <: CrossValGenerator
  function mkRandomSub(n_subjects::Int64, train_ratio::Float64=.8)
    RandomSub(n_subjects, round(Int64, train_ratio * n_subjects), n_folds)
  end
  @switch cvgT begin
    Kfold; n_subjects::Int64 -> Kfold(n_subjects, n_folds)
    RandomSub; mkRandomSub
  end
end

getCvg(cvgT::Type, n_folds::Int64, n_samples::Int64) = getCvgGen(cvgT, n_folds)(n_samples)


function learningCurve(o::Outcome;
                        dataset::DataSet=conn,
                        region::Region=left_select,
                        C=1.0,
                        subject_group::SubjectGroup=all_subjects,
                        seed::Nullable{Int}=Nullable(1234),
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.,
                        t::Target=diff_wpm
                        )
  isnull(seed) || srand(get(seed))

  svr = LinearSVR(C=C)
  lc_cvg_gen = getCvgGen(RandomSub, 25)
  learningCurve(svr, lc_cvg_gen, o, dataset, region, subject_group, train_ratios,
    t=t)
end


function learningCurve(di::DataInfo, C::Float64; seed::Nullable{Int}=Nullable(1234),
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.)
  learningCurve(di.outcome, region=di.region, C=C, subject_group=di.subject_group, seed=seed,
                 train_ratios=train_ratios; t=di.target)
end


function lcFlip(di::DataInfo, C::Float64, flip_ratio::Float64)
  ret = DataFrame()
  for C_tmp = [C/flip_ratio, C, C*flip_ratio]

    suff(k::Symbol) = if(C_tmp != C)
      C_tmp > C ? symbol(k, "_gt") : symbol(k, "_lt") else
      k end

    lc = learningCurve(di, C_tmp, train_ratios=5./6:1./6:1.)[end, :]
    ret[suff(:C)] = C_tmp

    for n in names(lc)
      ret[suff(n)] = lc[n]
    end

  end

  for f in fieldnames(di)
    ret[f] = di[f]
  end

  ret
end


macro set_seed()
  :(isnull(seed) || srand(get(seed)))
end


function ratiosToCounts(ratios::AbstractVector{Float64}, n::Int64)
  @assert all( (ratios .<= 1.) &  (ratios .>= 0.) )

  getCount(r::Float64) = round(Int, r*n)::Int64

  map(getCount, ratios)
end

ratioToCount(ratio::Float64, n::Int64) = ratiosToCounts([ratio], n)[1]


function samplesGen(cvg::CrossValGenerator,
                        sample_size::Int64,
                        is_perm=false)

  all_samples = 1:sample_size

  function shuffle(arr)
    shuffle_ixs = @>> arr length randperm(MersenneTwister())
    arr[shuffle_ixs]
  end

  typealias Inds Vector{Int64}
  typealias XyInds Tuple{Inds, Inds}
  typealias TrainTestInds Tuple{XyInds, XyInds}
  cached_ixs::Array{TrainTestInds} = map(cvg) do x_train_ixs
    x_test_ixs = setdiff(all_samples, x_train_ixs)

    y_train_ixs, y_test_ixs = if is_perm
      shuffle(x_train_ixs), shuffle(x_test_ixs)
    else
      x_train_ixs, x_test_ixs
    end

    trains::XyInds = (x_train_ixs, y_train_ixs)
    tests::XyInds = (x_test_ixs, y_test_ixs)

    TrainTestInds( (trains, tests) )
  end

  sample_ix::Int64 -> cached_ixs[sample_ix]
end


function crossValSamplesGen(cvg::CrossValGenerator,
  sample_size::Int64)

  all_samples = 1:sample_size

  typealias Inds Vector{Int64}
  cached_cross_val_ixs::Array{Tuple{Inds, Inds}} = map(cvg) do train_ixs
    test_ixs = setdiff(all_samples, train_ixs)
    (train_ixs, test_ixs)
  end

  repetition_ix::Int64 -> cached_cross_val_ixs[fit_ix]
end


function normalizeData(X::AbstractArray,
  feature_mns::AbstractVector, feature_stds::AbstractVector)

  @assert length(feature_mns) == length(feature_stds) == size(X, 2)

  (X .- feature_mns')./feature_stds'
end


updateStateMeanStdGen(state::Dict) = Xy::XY -> begin
  X, _ = Xy

  state[:feature_means] = mean(X, 1)[:]
  state[:feature_stds] = std(X, 1)[:]
  state[:variant_features] = state[:feature_stds] .> 1e-6

  Xy
end


doSomethingGen(fn, run) = run ? Xy::XY -> fn(Xy) : identity


normalizeDataGen(state::Dict, run) = doSomethingGen(run) do Xy::XY
  X, y = Xy

  norm_X = normalizeData(X, state[:feature_means], state[:feature_stds])

  norm_X, y
end


selectVariantColsGen(state::Dict, run) = doSomethingGen(run) do Xy::XY
  X, y = Xy
  variant_features = state[:variant_features]

  X[:, variant_features], y
end


updateStateBiasGen(state::Dict) = Xy::XY -> begin
  X, y = Xy

  state[:cors] = cor(X, y)
  state[:bias] = if sum(state[:cors] .< 0) > sum(state[:cors] .> 0)
    -1 * maximum(y)
  else
    -1 * minimum(y)
  end

  X, y
end


offsetBiasGen(state::Dict, run) = doSomethingGen(run) do Xy::XY
  Xy[1], Xy[2] + state[:bias]
end


reverseBiasGen(state::Dict, run) = y::Vector -> run ? y - state[:bias] : y


svrPipeline(di::DataInfo) = svrPipeline(getXyMat(di)...)


svrPipeline(X::Matrix, y::Vector) = classifierPipeline(X, y)


function classifierPipeline{T}(X::Matrix, y::Vector{T};
  normalize::Bool=false,
  select_variant_cols::Bool=true,
  classifier::PyObject = LinearSVR(),
  score_fn::Function = r2score
  )

  state = Dict{Symbol, Any}(:classifier => classifier, :C => classifier[:C])

  selectData(ixs::IXs; y_ixs::IXs=ixs) = X[ixs, :], y[y_ixs]

  selectVariantCols = selectVariantColsGen(state, select_variant_cols)

  normalizeData = normalizeDataGen(state, normalize)

  fit_fns::Functions = begin
    updateStateMeanStd! = updateStateMeanStdGen(state)

    function classifierFit!(Xy::XY)
      classifier[:C] = state[:C]
      curr_X::Matrix = Xy[1]
      curr_y::Vector{T} = Xy[2]
      classifier[:fit](curr_X, curr_y)
    end

    [selectData, updateStateMeanStd!,
      normalizeData, selectVariantCols,
      classifierFit!]
  end

  predict_fns::Functions = begin
    onlyX(Xy::XY) = Xy[1]
    mockY(X::Matrix) = (X, T[])

    normalizeXdata(X::Matrix) = X |> mockY |> normalizeData |> onlyX

    selectVariantColsX(X::Matrix) = X |> mockY |> selectVariantCols |> onlyX

    classifierPredict(X::Matrix) = classifier[:predict](X)

    [selectData, onlyX,
      normalizeXdata, selectVariantColsX,
      classifierPredict
      ]
  end

  Pipeline(fit_fns, predict_fns, score_fn, y, state)
end
