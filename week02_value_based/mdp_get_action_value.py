
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    # YOUR CODE HERE
    states = mdp.get_all_states()
    q = 0
    for s in states:
        q += mdp.get_transition_prob(state, action, s)*(mdp.get_reward(state, action, s) + gamma*state_values[s])

    return q
