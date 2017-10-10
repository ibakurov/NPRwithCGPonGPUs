package ec.cgp;


import ec.EvolutionState;
import ec.Individual;
import ec.Population;
import ec.Subpopulation;
import ec.cgp.representation.VectorIndividualCGP;
import ec.simple.SimpleBreeder;

/**
 * The CGP implementation requires this slightly modified SimpleBreeder. Its
 * sole purpose is to reset string representations of the expressions
 * represented by all genomes in the population. Resetting forces re-computation
 * of each expression during evaluation of CGP nodes, but only for the first
 * such evaluation.
 * 
 * @author David Oranchak, doranchak@gmail.com, http://oranchak.com
 * 
 */
@SuppressWarnings("serial")
public class Breeder extends SimpleBreeder {

	/**
	 * Reset the expressions that were computed and stored in the previous
	 * generation.
	 */
	public Population breedPopulation(EvolutionState state) {
		for (int x = 0; x < state.population.subpops.length; x++)
			for (int y = 0; y < state.population.subpops[x].individuals.length; y++)
				((VectorIndividualCGP) state.population.subpops[x].individuals[y]).expression = null;

		Population pop = super.breedPopulation(state);
		
		for (Subpopulation subpop : pop.subpops) {
			for(Individual indiv : subpop.individuals) {
				System.out.println(((VectorIndividualCGP)indiv).expression);
			}
		}
		return pop;
	}

}
