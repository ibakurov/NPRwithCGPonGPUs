package ec.multiobjective;

import java.util.Comparator;
import ec.Individual;


/** 
 * Compares any individuals with MultiObjectiveFitness fitness across a single objective
 * Individuals already implement the Comparable interface, if you want their sum of ranks or whatever.
 * @author      Michael Gircys <mg12vp@brocku.ca>
 * @version     23.1.0
 * @since       23.1.0
 */
public class MultiObjectiveIndividualComparator implements Comparator<Individual>
{	
	/** the objective index; which objective to compare. */
	public int obj_idx = 0;
	/** Should we be preferring descending order? Default sort is ascending. */
	public boolean desc = false; 
	
	/** 
	 * @param obj_idx		The index indicating which objective is to be compared
	 * @param descending	Indicates that a descending (as opposed to ascending) order is desired.
	 */
	public MultiObjectiveIndividualComparator(int obj_idx, boolean descending)
	{
		this.obj_idx = obj_idx;
		this.desc = descending;
	}
	
	@Override
	public int compare(Individual i1, Individual i2) 
	{
		MultiObjectiveFitness f1 = (MultiObjectiveFitness)(i1.fitness);
		MultiObjectiveFitness f2 = (MultiObjectiveFitness)(i2.fitness);
		double o1 = f1.objectives[obj_idx];
		double o2 = f2.objectives[obj_idx];
		
		if(o1==o2) return 0;
		
		boolean maximize = f1.maximize[obj_idx];
		boolean gt = o1 > o2;		// o1 greater value than o2
		gt = gt ^ (!maximize);		// o1 "better" value than o2 (if we're trying to minimize, is it lesser?)
		gt = gt ^ desc;				// If we're looking for "Descending" order, swap result.
		
		return gt ? 1 : -1;
	}
	
}