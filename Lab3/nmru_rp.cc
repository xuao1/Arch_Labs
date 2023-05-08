#include "mem/cache/replacement_policies/nmru_rp.hh"

#include <cassert>
#include <memory>
#include <vector>

#include "params/NMRURP.hh"
#include "sim/cur_tick.hh"

namespace gem5
{

GEM5_DEPRECATED_NAMESPACE(ReplacementPolicy, replacement_policy);
namespace replacement_policy
{

NMRU::NMRU(const Params &p)
  : Base(p)
{
}

void
NMRU::invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
{
    // Reset last touch timestamp
    std::static_pointer_cast<NMRUReplData>(
        replacement_data)->lastTouchTick = Tick(0);
}

void
NMRU::touch(const std::shared_ptr<ReplacementData>& replacement_data) const
{
    // Update last touch timestamp
    std::static_pointer_cast<NMRUReplData>(
        replacement_data)->lastTouchTick = curTick();
}

void
NMRU::reset(const std::shared_ptr<ReplacementData>& replacement_data) const
{
    // Set last touch timestamp
    std::static_pointer_cast<NMRUReplData>(
        replacement_data)->lastTouchTick = curTick();
}

ReplaceableEntry*
NMRU::getVictim(const ReplacementCandidates& candidates) const
{
    // There must be at least one replacement candidate
    assert(candidates.size() > 0);

    // Find the candidate with the biggest lastTouchTick
    auto max_lastTouchTick = std::static_pointer_cast<NMRUReplData>(candidates[0]->replacementData)->lastTouchTick;
    size_t max_index = 0;
    for (size_t i = 1; i < candidates.size(); ++i) {
        auto candidate_lastTouchTick = std::static_pointer_cast<NMRUReplData>(candidates[i]->replacementData)->lastTouchTick;
        if (candidate_lastTouchTick > max_lastTouchTick) {
            max_lastTouchTick = candidate_lastTouchTick;
            max_index = i;
        }
    }

    // Remove the choosen candidate
    std::vector<ReplaceableEntry*> new_candidates(candidates.begin(), candidates.end());
    new_candidates.erase(remaining_candidates.begin() + max_index);

    //Select a random entry from the remaining candidates
    ReplaceableEntry* victim = new_candidates[random_mt.random<unsigned>(0, new_candidates.size() - 1)];

    // Visit all candidates to search for an invalid entry. If one is found,
    // its eviction is prioritized
    for (const auto& candidate : candidates) {
        if (!std::static_pointer_cast<RandomReplData>(
                    candidate->replacementData)->valid) {
            victim = candidate;
            break;
        }
    }

    return victim;
}

std::shared_ptr<ReplacementData>
NMRU::instantiateEntry()
{
    return std::shared_ptr<ReplacementData>(new NMRUReplData());
}

} // namespace replacement_policy
} // namespace gem5
