using DataFrames
using Lazy
using Memoize
using MLBase
using PyCall

include("DataInfo.jl")
include("helpers.jl")

@pyimport sklearn.svm as svm
LinearSVR = svm.LinearSVR


macro l2_from_true(x)
  :(sum( (y_true - $x).^2))
end


function r2Score{T <: Real}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})

  numerator::Float64 = @l2_from_true y_pred
  denominator::Float64 = @l2_from_true mean(y_true)

  1 - numerator/denominator
end


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
                        score_fn::Function=r2Score)

  X, y = getXyMat(o, diff_wpm, dataset=dataset, region=region, subject_group=subject_group)
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
  @switch cvgT begin
    Kfold; (n_samples::Int64) -> Kfold(n_samples, n_folds)
    RandomSub; (n_samples::Int64) -> RandomSub(n_samples, round(Int64, .8 * n_samples), n_folds)
  end
end

getCvg(cvgT::Type, n_folds::Int64, n_samples::Int64) = getCvgGen(cvgT, n_folds)(n_samples)


function learningCurve(o::Outcome;
                        dataset::DataSet=conn,
                        region::Region=left_select,
                        C=1.0,
                        subject_group::SubjectGroup=all_subjects,
                        seed::Nullable{Int}=Nullable(1234),
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.)
  isnull(seed) || srand(get(seed))

  svr = LinearSVR(C=C)
  lc_cvg_gen = getCvgGen(RandomSub, 25)
  learningCurve(svr, lc_cvg_gen, o, dataset, region, subject_group, train_ratios)
end


function learningCurve(di::DataInfo, C::Float64; seed::Nullable{Int}=Nullable(1234),
                        train_ratios::AbstractVector{Float64}=1./6:1./6:1.)
  learningCurve(di.outcome, region=di.region, C=C, subject_group=di.subject_group, seed=seed,
                 train_ratios=train_ratios)
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


function getRepetitionSamplesGen(samples::AbstractVector,
  n_repetitions::Int64,
  is_perm::Bool)
  n_samples::Int64 = length(samples)

  typealias Inds Vector{Int64}
  cached_perm_ixs::Array{Inds} = map(1:n_repetitions) do r
    randperm(MersenneTwister(r), n_samples)
  end

  getPermInds(repetition_ix, train_size) =(
    cached_perm_ixs[repetition_ix][1:train_size],
    cached_perm_ixs[repetition_ix][train_size+1:end]
  )

  typealias TrainTest Tuple{Inds, Inds}
  cvg::CrossValGenerator = getCvg(RandomSub, n_repetitions, n_samples)
  cached_pure_ixs::Array{TrainTest} = map(cvg) do train_inds::Inds
    train_inds, setdiff(1:n_samples, train_inds)
  end

  function fn(repetition_ix::Int64)
    (X_train::Inds, X_test::Inds) = cached_pure_ixs[repetition_ix]
    (y_train::Inds, y_test::Inds) = if is_perm
      ret = getPermInds(repetition_ix, length(X_train))
      #TODO: Figure out why this is necessary
      ret
    else
      X_train, X_test
    end

    (samples[X_train], samples[y_train]), (samples[X_test], samples[y_test])
  end
end
