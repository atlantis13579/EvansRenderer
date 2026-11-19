#include "Renderer/Lighting/SubsurfaceDiffusion.h"

#include <limits>

namespace Evans::Renderer
{
    namespace
    {
        // Christensen and Burley, "Approximate Reflectance Profiles for
        // Efficient Subsurface Scattering", Pixar Technical Memo 15-04 (2015).
        // https://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf
        // R(r)=(exp(-r/d)+exp(-r/(3d)))/(8*pi*d*r). After multiplying
        // by the polar area element, its radial CDF is a two-exponential mixture.
        constexpr double Pi = 3.14159265358979323846;
        constexpr double LongLobeScale = 3;
        constexpr double ShortLobeWeight = 1 / (1 + LongLobeScale);
        constexpr double LongLobeWeight = LongLobeScale / (1 + LongLobeScale);
        constexpr double SupportTailProbability = 0.001;
        constexpr std::size_t AngularSamples = 64;
        double RadialCdf(double radius, double distance)
        {
            if (radius <= 0) return 0;
            return -ShortLobeWeight * std::expm1(-radius / distance) -
                LongLobeWeight * std::expm1(-radius / (LongLobeScale * distance));
        }
        double BinMass(double left, double right, double distance)
        {
            double mass = 0;
            for (std::size_t sample = 0; sample < AngularSamples; ++sample)
            {
                const double cosine = std::cos(2 * Pi * (sample + 0.5) / AngularSamples);
                double nearRadius, farRadius;
                if (cosine > 0)
                {
                    if (right <= 0) continue;
                    nearRadius = std::max(left, 0.0) / cosine; farRadius = right / cosine;
                }
                else
                {
                    if (left >= 0) continue;
                    nearRadius = std::min(right, 0.0) / cosine; farRadius = left / cosine;
                }
                mass += std::max(RadialCdf(farRadius, distance) - RadialCdf(nearRadius, distance), 0.0);
            }
            return mass / AngularSamples;
        }
    }
    SubsurfaceKernel BuildSubsurfaceKernel(const Techniques::SubsurfaceScatteringSettings& requested)
    {
        const auto settings = Techniques::SanitizeSubsurfaceScatteringSettings(requested);
        SubsurfaceKernel result;
        const double maximumRadius = *std::max_element(settings.ChannelRadius.begin(), settings.ChannelRadius.end());
        const double support = -LongLobeScale * maximumRadius * std::log(SupportTailProbability);
        result.Support = static_cast<float>(support);
        std::array<double, SubsurfaceKernelSize> positions;
        for (std::size_t tap = 0; tap < positions.size(); ++tap)
        {
            const double unit = double(tap) * 2 / (positions.size() - 1) - 1;
            positions[tap] = support * unit * unit * unit;
            result.Taps[tap].W = static_cast<float>(positions[tap]);
        }
        const double infinity = std::numeric_limits<double>::infinity();
        for (std::size_t channel = 0; channel < settings.ChannelRadius.size(); ++channel)
        {
            std::array<double, SubsurfaceKernelSize> weights;
            double total = 0;
            for (std::size_t tap = 0; tap < positions.size(); ++tap)
            {
                const double left = tap ? (positions[tap - 1] + positions[tap]) / 2 : -infinity;
                const double right = tap + 1 < positions.size() ? (positions[tap] + positions[tap + 1]) / 2 : infinity;
                weights[tap] = BinMass(left, right, settings.ChannelRadius[channel]); total += weights[tap];
            }
            float roundedTotal = 0;
            for (std::size_t tap = 0; tap < positions.size(); ++tap)
            {
                const float weight = static_cast<float>(weights[tap] / total);
                auto& value = result.Taps[tap];
                if (channel == 0) value.X = weight; else if (channel == 1) value.Y = weight; else value.Z = weight;
                roundedTotal += weight;
            }
            // Correct only float conversion roundoff; the integrated profile's
            // energy is one in every color channel, including its finite tails.
            auto& center = result.Taps[SubsurfaceKernelSize / 2];
            if (channel == 0) center.X += 1 - roundedTotal;
            else if (channel == 1) center.Y += 1 - roundedTotal; else center.Z += 1 - roundedTotal;
        }
        return result;
    }
}
