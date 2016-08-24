using DataFrames
using Lazy
using StatsBase


function randomizationTest(compare_fn,
  truths::AbstractVector, perms::AbstractVector;
  num_tests=1000)

  concats = [truths; perms]
  num_samples = length(concats)
  num_truths = length(truths)

  actual = compare_fn(truths, perms)

  tests = map(1:num_tests) do i
    sampled_data = @> concats sample(num_samples, replace=false)
    compare_fn(sampled_data[1:num_truths], sampled_data[num_truths+1:end])
  end

  p_pos = sum([s <= actual for s in tests])/num_tests

  p_pos, actual, tests
end


function diffInMeans(truths::AbstractVector, perms::AbstractVector;
  num_tests = 1000)

  meanDiff(arr1, arr2) = mean(arr1) - mean(arr2)

  randomizationTest(meanDiff, truths, perms)
end


function diffInMeans(real_coefs::DataFrame, perm_coefs::DataFrame;
  num_tests=1000)

  predictors = names(real_coefs)
  @assert predictors == names(perm_coefs)

  p_pos = map(predictors) do p
    rank, _, _ = diffInMeans(real_coefs[p], perm_coefs[p], num_tests=num_tests)
    rank
  end

  @> DataFrame(predictor=predictors, p_pos=p_pos, p_neg=1 .- p_pos) sort(cols=:p_pos)
end


function diffInMeans(coefs::DataFrame;
  num_tests=1000)

  perm_coefs = begin
    num_samples = size(coefs, 1)
    reduce(DataFrame(), names(coefs)) do df, p
      df[p] = zeros(Float64, num_samples)
      df
    end
  end

  diffInMeans(coefs, perm_coefs, num_tests=num_tests)
end


function permutationPvalAvg(reals, perms; agg=mean)
  sorted_perms = sort(perms)
  n_perms = length(perms)

  rank(r, ix=1) = if (ix > n_perms) || (r < sorted_perms[ix])
    ix - 1
  else
    rank(r, ix+1)
  end

  @>> reals map(r -> (rank(r) + 1)/(n_perms + 1)) agg
end
