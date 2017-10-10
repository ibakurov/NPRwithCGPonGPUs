package ec.multiobjective;

import ec.EvolutionState;
import ec.Individual;
import ec.simple.SimpleEvaluator;
import ec.util.Parameter;


/**
 * Evaluator for SumOfRanksFitness
 * Ranks individuals based on their objectives and produces an aggregate (normalized) sum of ranks fitness value.
 * @author      Michael Gircys <mg12vp@brocku.ca>
 * @version     23.1.0
 * @since       23.1.0
 */
@SuppressWarnings("serial")
public class SumOfRanksEvaluator extends SimpleEvaluator
{
	
	@Override
    public void setup(final EvolutionState state, final Parameter base)
    {
        super.setup(state,base);
    } 

	@Override
	public void evaluatePopulation(final EvolutionState state)
    {
		// Produce phenotype, run trials / extract features, produce base raw and objective scores
		super.evaluatePopulation(state);
		
		// Then we rank, weight, and sum them.
		for(int s = 0; s < state.population.subpops.length; s++)
		{	
			Individual[] inds = state.population.subpops[s].individuals;
			
			// only rank subpopulations using SumOfRanksFitness
			if (!(inds[0].fitness instanceof SumOfRanksFitness)) return;
			
			SumOfRanksFitness.RankObjectives(inds);
		}
    }
	
}
