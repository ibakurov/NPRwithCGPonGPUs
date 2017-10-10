package ec.multiobjective;

import java.util.Comparator;
import ec.EvolutionState;
import ec.Fitness;
import ec.Individual;
import ec.util.Parameter;

/**
 * Fitness class used to find the (normalized) sum of ranks multiobjective fitness
 * Based loosely on Rank.java by Bergen and Harrington
 * @author      Michael Gircys <mg12vp@brocku.ca>
 * @version     23.1.0
 * @since       23.1.0
 */
@SuppressWarnings("serial")
public class SumOfRanksFitness extends MultiObjectiveFitness 
{
	// Example:
	/*
	
	 * Config:
	 		eval                                        = ec.multiobjective.SumOfRanksEvaluator
	 		pop.subpop.0.species.fitness                = ec.multiobjective.SumOfRanksFitness
			pop.subpop.0.species.fitness.showraw        = true
			pop.subpop.0.species.fitness.normalize      = true
			pop.subpop.0.species.fitness.num-objectives = 2
			pop.subpop.0.species.fitness.0.name         = Some Error Measure
			pop.subpop.0.species.fitness.0.weight       = 1
			pop.subpop.0.species.fitness.0.maximize     = false
			pop.subpop.0.species.fitness.1.name         = Some Hits Measure
			pop.subpop.0.species.fitness.1.weight       = 1
			pop.subpop.0.species.fitness.1.maximize     = true
			
	 * Problem:
	 		@Override
			public void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum) 
			{
				// ...
				SumOfRanksFitness f = ((SumOfRanksFitness)ind.fitness);
        		f.raws[0] = calc_measure();		// raw measure
        		f.raws[1] = calc_hits();		// raw hits
        		f.setObjectives(state, new double[]
        				{  
        					Math.abs( f.raws[0] - target_measure  ),	// error (distance from target)
        					f.raws[1] 									// hits
        				});
        		ind.evaluated = true;
			}
	 	
	 */

	// Common:
	// These should be "shared"/common between instances (across a subpop).
	
	/** The names or labels for each of the objectives; configured at setup */
	public String[] names;
	/** The weight each objective should have; configured at setup; defaults to 1.0 */
	public double[] weights;
	/** The maximum rank assigned for each objective during the last evaluation; intermediate computation; set by SumOfRanksEvaluator */
	public int[]    maxRank;
	/** the accumulating diversity penalty; should be positive value; configured at setup */
	public double   penalty_diversity;
	/** normalize the sum of ranks?; configured at setup */
	public boolean  normalize;
	/** display raw values instead of objectives for human readable fitness?; configured at setup */
	public boolean  showraw;
	
	// Unique:
	// These should be unique to each individual
	
	/** aggregate weighted sum of ranks; set by SumOfRanksEvaluator */
	public double   fitness;
	/** any diversity or other penalties; set by SumOfRanksEvaluator */
	public double   penalties; 
	/** the rank of each objective relative to the last population evaluated; set by SumOfRanksEvaluator */
	public double[] ranks;
	/** the raw (signed) objective values for charting / human interpretation; you should provide these during evaluation. */
	public double[] raws;
	
	
	public static final String P_NAMES = "name";
	public static final String P_WEIGHTS = "weight";
	public static final String P_PENALTIES = "penalty";
	public static final String P_PENALTY_DIVERSITY = "diversity";
	public static final String P_NORMALIZE = "normalize";
	public static final String P_SHOWRAW = "showraw";

	
	public String[] getNames()     { return names;    }
	public String   getName(int i) { return names[i]; }
	public String[] getAuxilliaryFitnessNames() { return getNames(); }	
	public double[] getWeights()     { return weights;    }
	public double   getWeight(int i) { return weights[i]; }
	
	
	public Object clone()
	{
		SumOfRanksFitness f = (SumOfRanksFitness)(super.clone());
		f.ranks = (double[])(ranks.clone());
		f.raws  = (double[])(raws.clone());
		// From MultiObjectiveFitness:
		// 		"note that we do NOT clone max and min fitness, or maximizing -- they're shared"
		// so we're not doing anything for our new stuff either, since it's common.
		return f;
	}
	
	public void setup(EvolutionState state, Parameter base)
	{
		super.setup(state, base);
		Parameter def = defaultBase();
		int numFitnesses = getNumObjectives();
        
        names        = new String[numFitnesses];
		weights      = new double[numFitnesses];
		maxRank      = new int[numFitnesses];
		ranks        = new double[numFitnesses];
        raws         = new double[numFitnesses];
		
		for (int i = 0; i < numFitnesses; i++)
		{
			Parameter objective    = base.push(""+i);
			Parameter objectivedef = base.push(""+i);
			
			// load default globals
			names[i]    = state.parameters.getStringWithDefault( base.push(P_NAMES),    def.push(P_NAMES),    "Unknown Objective" );
			weights[i]  = state.parameters.getDoubleWithDefault( base.push(P_WEIGHTS),  def.push(P_WEIGHTS),  1.0 );
			maximize[i] = state.parameters.getBoolean(           base.push(P_MAXIMIZE), def.push(P_MAXIMIZE), maximize[i] );
			// load specifics if any
			names[i]    = state.parameters.getStringWithDefault( objective.push(P_NAMES),    objectivedef.push(P_NAMES),    names[i] );
			weights[i]  = state.parameters.getDoubleWithDefault( objective.push(P_WEIGHTS),  objectivedef.push(P_WEIGHTS),  weights[i] );
			maximize[i] = state.parameters.getBoolean(           objective.push(P_MAXIMIZE), objectivedef.push(P_MAXIMIZE), maximize[i] );
		}
		
		penalty_diversity = state.parameters.getDoubleWithDefault( base.push(P_PENALTIES).push(P_PENALTY_DIVERSITY), null, 0.0);
		normalize = state.parameters.getBoolean( base.push(P_NORMALIZE), null, true);
		showraw = state.parameters.getBoolean( base.push(P_SHOWRAW), null, false);
		
		state.output.exitIfErrors();
	}
	
	
	public double[] getRanks()                { return ranks;    }
	public double   getRanks(int i)           { return ranks[i]; }
	public void     setRanks(int i, double r) { ranks[i] = r;    }
	public void     setRanks(double[] r)      { ranks = r;       }
	
	
	/** returns the aggregate normalized sum of ranks (after being ranked by SumOfRanksEvaluator) */
	public double fitness()            { return fitness; }
	public double getFitness()         { return fitness; }
	public void   setFitness(double f) { fitness = f;    }
	
	public boolean isIdealFitness()
	{	
		return false; // Something ranked 1st in all objectives may still not be 'ideal'.
	}
	/** returns true if our aggregate (normalized) sum of ranks is equivalent */
	public boolean equivalentTo(Fitness _fitness)
    {
	    return this.fitness() == _fitness.fitness();
	}
	/** returns true if our aggregate (normalized) sum of ranks is better */
	public boolean betterThan(Fitness _fitness)
	{
		return this.fitness() >  _fitness.fitness();
	}
	
	
	public String fitnessToStringForHumans()
	{
		String s = FITNESS_PREAMBLE + MULTI_FITNESS_POSTAMBLE;
		for (int x = 0; x < objectives.length; x++)
			s  +=  ( x>0 ? " " : "" )  +  ( showraw ? raws[x] : objectives[x] )  ;
		s += FITNESS_POSTAMBLE;
		return s;
	}
	
	/** 
	 * Recomputes the individuals fitness 
	 * assuming correct and updated ranks[] and maxRank[] information 
	 **/
	public void recomputeFitness()
	{
		fitness = 0.0;
		double totalweight = 0.0;
		
		// Calculate normalized ranks
		for ( int o = 0; o < objectives.length; o++ ) 
		{
			double weight   = weights[o];
			double objScore = ( maxRank[o] - ranks[o] ) * weight;
			
			if(normalize) objScore /= maxRank[o];
			
			fitness     += objScore;
			totalweight += weight;  
		}
		
		if(normalize) fitness /= totalweight;
		
		fitness = Math.max( 0.0, fitness - penalties );
	}
	
	
	
	
	
	// Below, adapted from Rank.java by Bergen and Harrington
	
	
	/** 
	 * Method which is called to rank a population of individuals based on fitness values for all objectives.
	 * @param pop               Population to rank.
	 */
	public static void RankObjectives ( Individual[] pop ) 
	{
		// prototypical SumOfRanksFitness instance
		SumOfRanksFitness fit0 = ((SumOfRanksFitness)(pop[0].fitness));
		
		// For each fitness function, sort population and set ranks
		int length = fit0.getNumObjectives();
		for ( int i = 0; i < length; i++ ) 
		{
			pop = Sort( pop, i );
			pop = Rank( pop, i );
		}

		// Diversify
		if(fit0.penalty_diversity > 0.0)
			AddDiversityPenalties( pop );
		
		// Set fitness values (rank) to population
		for( int i = 0; i < pop.length; i++ )
			((SumOfRanksFitness)(pop[i].fitness)).recomputeFitness();
	}

	/**
	 * Adds accumulating diversity penalties to chains of individuals with identical fitness.
	 * @param pop	The sorted population to check for equality
	 */
	public static void AddDiversityPenalties(Individual[] pop)
	{
		int currentChain  = 0;
		double accPenalty = 0.0;
		
		// population already sorted by the last objective,
		// and equality requires all objectives equal between individuals
		for(int i = 1; i < pop.length; i++)
		{
			SumOfRanksFitness fit1 = (SumOfRanksFitness)(pop[i].fitness);
			SumOfRanksFitness fit2 = (SumOfRanksFitness)(pop[i-1].fitness);
			
			currentChain = (IsEqual(fit1.objectives, fit2.objectives)) ? currentChain+1 : 0;			
			accPenalty   = currentChain * fit1.penalty_diversity;
			fit1.penalties = accPenalty;
		}
	}

	/**
	 * Equality requires the same amount of objectives, and all objectives equal
	 * @param set1	first  compared set of objectives
	 * @param set2	second compares set of objectives
	 * @return		are all objectives equal 
	 */
	public static boolean IsEqual(double[] set1, double[] set2)
	{
		if(set1.length != set2.length) return false;
		for(int i = 0; i < set1.length; i++)
			if(set1[i] != set2[i])
				return false;
		return true;
	}
	
	/**
	 * Sorts the population based on an objective.
	 * Sorted best to worst (rank 0 to ~n). 
	 * @param obj_idx           Index of the objective
	 * @param pop               Population
	 * @param f                 Fitness function
	 * @return                  Sorted population
	 */
	public static Individual[] Sort (Individual[] pop, int obj_idx) 
	{
		// Original implementation used bubblesort? Bergen, why?
		// I had updated it as an intermediate step, but there's better ways.
		/*
		boolean swapped = true;
		while (swapped)
		{
			swapped = false;
			for (int i = 0; i < pop.length - 1; i++)
			{
				SumOfRanksFitness fit1 = (SumOfRanksFitness)(pop[i].fitness);
				SumOfRanksFitness fit2 = (SumOfRanksFitness)(pop[i+1].fitness);
				if ( ( fit1.objectives[obj_idx] > fit2.objectives[obj_idx] &&  fit1.maximize[obj_idx] ) || 
					 ( fit1.objectives[obj_idx] < fit2.objectives[obj_idx] && !fit1.maximize[obj_idx] )  )
				{
					Individual temp = pop[i];
					pop[i] = pop[i+1];
					pop[i+1] = temp;
					swapped = true;
				}
			}
		}
		*/
		
		// Threading overhead is is not worth it for less than ~2500 items.
		// https://www.javacodegeeks.com/2013/04/arrays-sort-versus-arrays-parallelsort.html
		// TODO: Verify. This was originally done for int[] which has a simple comparator.
		// TODO: Investigate ec.util SortArray. Advantages?
		// Still, better than BubbleSort.
		Comparator<Individual> cmp = new MultiObjectiveIndividualComparator(obj_idx,true);
		if(pop.length > 2500)
				java.util.Arrays.parallelSort(pop, cmp);
		else	java.util.Arrays.sort(pop, cmp);
		return pop;
	}

	/**
	 * Set the ranks of a sorted population.
	 * Lower rank is "better".
	 * @param obj_idx           Index of the objective
	 * @param pop               Population
	 * @return                  Ranked population
	 */
	public static Individual[] Rank ( Individual[] pop, int obj_idx ) 
	{
		int    rank    = 0;
		double lastVal = 0.0;
		
		for (int i = 0; i < pop.length; i++)
		{
			SumOfRanksFitness fit = ((SumOfRanksFitness)(pop[i].fitness));
			
			// The rank should not be updated in case of ties
			if( i > 0 && fit.objectives[obj_idx] != lastVal )
			{
				lastVal = fit.objectives[obj_idx];
				rank++;
			}
			
			fit.ranks[obj_idx] = rank;			
		}
		
		// Update the max rank record for the current objective
		// maxRank is a common/shared instance between SumOfRanksFitness objects 
		((SumOfRanksFitness)(pop[0].fitness)).maxRank[obj_idx] = rank;
		
		return pop;
	}
	
}
