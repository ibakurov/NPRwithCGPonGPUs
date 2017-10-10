package ec.gp.cuda;

import ec.EvolutionState;
import ec.simple.SimpleFinisher;

/**
 * A finisher which will make sure to deallocate all CUDA allocations and 
 * clean up after itself!
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaFinisher extends SimpleFinisher {

	private static final long serialVersionUID = 3602688663093867976L;

	@Override
	public void finishPopulation(EvolutionState state, int result) {
//		CudaEvolutionState cuState = (CudaEvolutionState) state;
//		cuState.getActiveJob().freeAll();
	}
	
}
