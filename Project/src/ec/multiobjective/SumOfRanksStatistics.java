package ec.multiobjective;

import ec.EvolutionState;
import ec.simple.SimpleStatistics;
import ec.util.*;


/**
 * Statistics for MultiObjective fitness schemes.
 * Requires SumOfRanksFitness, since raw values are not included in the MultiObjectiveFitness base type.
 * For each generation:
 *  	Prints the min, max, and average of each objective;
 *  	Prints the min, max, and average of each raw value;
 *  	Prints the "best" raw value found.
 * @author      Michael Gircys <mg12vp@brocku.ca>
 * @version     23.1.0
 * @since       23.1.0
 */
@SuppressWarnings("serial")
public class SumOfRanksStatistics extends SimpleStatistics
{   
    
	@Override
    public void setup(final EvolutionState state, final Parameter base)
    {
    	super.setup(state,base);
    }
    
	@Override
	public void postInitializationStatistics(final EvolutionState state)
	{
		super.postInitializationStatistics(state);
		
		if(state.population.subpops.length > 1)
    		state.output.fatal("" + this.getClass() + " does not support multiple subpopulations.");
	}
	
    @Override
    public void finalStatistics(final EvolutionState state, final int result)
    {   
    	// We don't want any final statistics from SimpleStatistics to be printed
    	//super.finalStatistics(state,result);
    }
    
    @Override
    public void postEvaluationStatistics(final EvolutionState state)
    {
    	SumOfRanksFitness typicalFitness = (SumOfRanksFitness)(state.population.subpops[0].individuals[0].fitness);
    	
    	int objs = typicalFitness.getNumObjectives();
    	int inds = state.population.subpops[0].individuals.length;
    	double[] obj_min = new double[objs];
    	double[] obj_max = new double[objs];
    	double[] obj_avg = new double[objs];
    	double[] raw_min = new double[objs];
    	double[] raw_max = new double[objs];
    	double[] raw_avg = new double[objs];
    	double[] raw_bst = new double[objs];
    	for(int o = 0; o < objs; o++)
    	{
    		obj_min[o] = Double.POSITIVE_INFINITY;
			raw_min[o] = Double.POSITIVE_INFINITY;
			obj_max[o] = Double.NEGATIVE_INFINITY;
			raw_max[o] = Double.NEGATIVE_INFINITY;
			obj_avg[o] = 0.0;
			raw_avg[o] = 0.0;
			raw_bst[o] = Double.NaN;
    	}
    	
    	// Get Aggregates
    	for(int i = 0; i < inds; i++)
    	{
    		SumOfRanksFitness fit = (SumOfRanksFitness)(state.population.subpops[0].individuals[i].fitness);
    		for(int o = 0; o < objs; o++)
        	{
    			obj_min[o] = Math.min(obj_min[o], fit.objectives[o]);
    			raw_min[o] = Math.min(raw_min[o], fit.raws[o]);
    			obj_max[o] = Math.max(obj_max[o], fit.objectives[o]);
    			raw_max[o] = Math.max(raw_max[o], fit.raws[o]);
    			obj_avg[o] += fit.objectives[o];
    			raw_avg[o] += fit.raws[o];
    			
    			if( ( fit.maximize[o] && obj_max[o] == fit.objectives[o]) ||
    				(!fit.maximize[o] && obj_min[o] == fit.objectives[o]) )
    				raw_bst[o] = fit.raws[o];
        	}
    	}
    	for(int o = 0; o < objs; o++)
    	{
    		obj_avg[o] /= inds;
        	raw_avg[o] /= inds;
    	}
    	
    	// Output
    	state.output.print("" + state.generation + " ", statisticslog);
    	for(int o = 0; o < objs; o++)
    	{	
    		String s = "[";
    		s += " " + obj_min[o];
    		s += " " + obj_avg[o];
    		s += " " + obj_max[o];
    		s += " " + raw_min[o];
    		s += " " + raw_avg[o];
    		s += " " + raw_max[o];
    		s += " " + raw_bst[o];
    		s += " ] ";
    		state.output.print(s, statisticslog);
    	}
    	state.output.println("", statisticslog);
    	
    }
    
}
