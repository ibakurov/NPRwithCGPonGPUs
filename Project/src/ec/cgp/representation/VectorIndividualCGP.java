package ec.cgp.representation;

import ec.vector.VectorIndividual;


/**
 * Base class for integer- and float-based CGP individuals.
 * 
 * @author David Oranchak, doranchak@gmail.com, http://oranchak.com
 * 
 */
public abstract class VectorIndividualCGP extends VectorIndividual {

	private static final long serialVersionUID = -4721258207969155736L;
	/** Temporary storage for displaying the full program */
	public StringBuffer expression;
	
	/** Return the genome. */
	public abstract Object getGenome();

}
