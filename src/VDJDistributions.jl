module VDJDistributions
    using Turing
    using Random
    using Distributions

    const eps = 1e-6

    export ShiftedPoisson, ZISNB, MixtureZISNB, BiZISNB, eps

    """
        ShiftedPoisson{T}(λ, shift)

    Defines the Shifted Poisson distribution.
    - `λ`: The rate parameter of the Poisson distribution.
    - `shift`: The integer value by which the Poisson distribution is shifted.
    """
    struct ShiftedPoisson{T<:Real} <: DiscreteUnivariateDistribution
        λ::T
        shift::Int
    end

    """
        Base.rand(d::ShiftedPoisson)

    Generates a random sample from the Shifted Poisson distribution.
    """
    Base.rand(d::ShiftedPoisson) = rand(Poisson(d.λ)) + d.shift

    """
        Distributions.pdf(d::ShiftedPoisson, x::Int)

    Calculates the probability density function for the Shifted Poisson distribution at a given integer `x`.
    """
    Distributions.pdf(d::ShiftedPoisson, x::Int) = pdf(Poisson(d.λ), x - d.shift)

    """
        Distributions.logpdf(d::ShiftedPoisson, x::Int)

    Calculates the log of the probability density function for the Shifted Poisson distribution at a given integer `x`.
    """
    Distributions.logpdf(d::ShiftedPoisson, x::Int) = logpdf(Poisson(d.λ), x - d.shift)

    """
        ZISNB{T}(π, r, p)

    Defines the Zero-Inflated Shifted Negative Binomial distribution.
    - `π`: Probability of zero inflation.
    - `r`: Number of successes for the negative binomial component.
    - `p`: Probability of success for the negative binomial component.
    """
    struct ZISNB{T<:Real} <: Distributions.DiscreteUnivariateDistribution
        π::T
        r::T
        p::T
    end

    """
        Distributions.rand(rng::AbstractRNG, d::ZISNB)

    Generates a random sample from the Zero-Inflated Shifted Negative Binomial distribution.
    """
    function Distributions.rand(rng::AbstractRNG, d::ZISNB)
        if rand(rng) < d.π
            return 0
        else
            return rand(rng, NegativeBinomial(d.r, d.p)) + 1
        end
    end

    """
        Distributions.logpdf(d::ZISNB, x::Int)

    Calculates the log of the probability density function for the Zero-Inflated Shifted Negative Binomial distribution at a given integer `x`.
    """
    function Distributions.logpdf(d::ZISNB, x::Int)
        if x == 0
            return log(d.π)
        elseif x > 0 
            return log1p(-d.π) + logpdf(NegativeBinomial(d.r, d.p), x - 1)
        else
            return -Inf
        end
    end

    """
        MixtureZISNB{T}(zisnb, negative_min, negative_max, mixture_prob)

    Defines the MixtureZISNB distribution that combines the ZISNB and DiscreteUniform distributions for negative values.
    - `zisnb`: The Zero-Inflated Shifted Negative Binomial component.
    - `negative_min`: Minimum value for the negative range.
    - `negative_max`: Maximum value for the negative range.
    - `mixture_prob`: Probability of selecting the ZISNB component.
    """
    struct MixtureZISNB{T<:Real} <: Distributions.DiscreteUnivariateDistribution
        zisnb::ZISNB{T}
        negative_min::Int
        negative_max::Int
        mixture_prob::T
    end

    """
        Distributions.rand(rng::AbstractRNG, d::MixtureZISNB)

    Generates a random sample from the MixtureZISNB distribution.
    """
    function Distributions.rand(rng::AbstractRNG, d::MixtureZISNB)
        if rand(rng) < d.mixture_prob
            return rand(rng, d.zisnb)
        else
            return rand(rng, DiscreteUniform(d.negative_min, d.negative_max))
        end
    end

    """
        Distributions.logpdf(d::MixtureZISNB, x::Real)

    Calculates the log of the probability density function for the MixtureZISNB distribution at a given value `x`.
    """
    function Distributions.logpdf(d::MixtureZISNB, x::Real)
        if x >= 0
            return log(d.mixture_prob) + logpdf(d.zisnb, x)
        elseif x >= d.negative_min && x <= d.negative_max
            return log1p(-d.mixture_prob) + logpdf(DiscreteUniform(d.negative_min, d.negative_max), x)
        else
            return -Inf
        end
    end

    """
        BiZISNB{T1, T2, L}(π_left, p_left, π_right, p_right, α, β, len)

    Defines the BiZISNB distribution that combines two ZISNB distributions.
    - `π_left`: Probability of zero inflation for the left trimming.
    - `p_left`: Probability of success for the left trimming.
    - `π_right`: Probability of zero inflation for the right trimming.
    - `p_right`: Probability of success for the right trimming.
    - `α`: Coefficient for the exponential part of the distribution.
    - `β`: Coefficient for the length-dependent part of the distribution.
    - `len`: Length parameter.
    """
    struct BiZISNB{T1 <: Real, T2 <: Real, L <: Real} <: Distribution{Multivariate, Discrete}
        π_left::T1
        p_left::T1
        π_right::T2
        p_right::T2
        α::T1
        β::T1
        len::L
    end

    """
        Distributions.rand(rng::AbstractRNG, d::BiZISNB)

    Generates a random sample from the BiZISNB distribution.
    """
    function Distributions.rand(rng::AbstractRNG, d::BiZISNB)
        mu = exp(d.α + d.β * d.len) + eps
        left_trimming = rand(rng, ZISNB(d.π_left, mu, d.p_left))
        right_trimming = rand(rng, ZISNB(d.π_right, mu, d.p_right))
        return (left_trimming, right_trimming)
    end

    """
        Distributions.rand(rng::AbstractRNG, d::BiZISNB, n::Int)

    Generates `n` random samples from the BiZISNB distribution.
    """
    function Distributions.rand(rng::AbstractRNG, d::BiZISNB, n::Int)
        return [rand(rng, d) for _ in 1:n]
    end

    """
        Distributions.logpdf(d::BiZISNB, x::Tuple{Int, Int})

    Calculates the log of the probability density function for the BiZISNB distribution at a given tuple `x`.
    """
    function Distributions.logpdf(d::BiZISNB, x::Tuple{Int, Int})
        left_trimming, right_trimming = x
        mu = exp(d.α + d.β * d.len) + eps
        logpdf_left = logpdf(ZISNB(d.π_left, mu, d.p_left), left_trimming)
        logpdf_right = logpdf(ZISNB(d.π_right, mu, d.p_right), right_trimming)
        return logpdf_left + logpdf_right
    end

    """
        Distributions.loglikelihood(d::BiZISNB, x::Tuple{Int, Int})

    Calculates the log likelihood for the BiZISNB distribution at a given tuple `x`.
    """
    function Distributions.loglikelihood(d::BiZISNB, x::Tuple{Int, Int})
        return logpdf(d, x)
    end
end