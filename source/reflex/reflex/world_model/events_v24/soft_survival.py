"""Expected original one-shot terminal cost under a learned soft hazard."""



def expected_terminal_reward(rewards, probability, index, cost):
    # Outer c_t supplies survival until this step. c_t * p_t is the
    # probability mass of FIRST termination at this step, not a repeated fee.
    if index is None:
        return rewards
    result = rewards.clone()
    result[..., index] = probability * cost
    return result
