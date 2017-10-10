package ec.cgp.problems;

import ec.EvolutionState;
import ec.Individual;
import ec.cgp.Evaluator;
import ec.cgp.Record;
import ec.cgp.representation.VectorIndividualCGP;
import ec.gp.cuda.CudaData;
import ec.gp.cuda.CudaProblem;
import ec.util.Parameter;
import main.NPRWithCGPOnGPUsProblem;

/**
 * An extension of CGPProblem which provides some facilities to represent
 * and evaluate classification problems.
 * 
 * @author David Oranchak, doranchak@gmail.com, http://oranchak.com
 * 
 */
@SuppressWarnings("serial")
public abstract class ClassificationProblem extends CudaProblem {
    
	/**
	 * Configure this Classification Problem. 
	 */
	public void setup(EvolutionState state, Parameter base) {
		super.setup(state, base);
	}
	
	
	/**
	 * Evaluate this individual. Fitness is set to the proportion of
	 * unsuccessfully classified training instances. If there are constant
	 * values, the input vector is filled with them starting at the end.
	 * 
	 * The test set is also evaluated to measure performance of the classifier.
	 */
	public abstract void evaluate(EvolutionState state, Individual ind, int subpopulation, int threadnum);

	/**
	 * Your implementing class must map attributes of the given Record to items
	 * in the input vector using this method.
	 * 
	 * @param inputs
	 *            input array to set
	 * @param r
	 *            record to set values from
	 */
	protected abstract void setInputs(Object[] inputs, Record r);

	/**
	 * During evaluation, outputs are passed to this method. Your implementing
	 * class must compare the outputs to the target class(es) in the given
	 * Record. Each element of the return vector is set to true if the
	 * corresponding output represents a successfully classified instance.
	 * 
	 * @param outputs The output vector resulting from evaluation of the CGP.
	 * @param r The record from which to compare classification results.
	 * @return A boolean vector indicating sucess of classification.
	 */
	//abstract double[] compare(Float[] outputs, Record r);

	/** 
	 * Sets the inputs, runs the Cartesian Genetic Program, and returns the results
	 * of classification.
	 * 
	 * @param state The evolution state
	 * @param threadnum The current thread number
	 * @param inputs The input vector
	 * @param rec The current record
	 * @param ind The current individual
	 * @return boolean vector indicating successful/unsuccessful classification(s).
	 */
	protected Float[] eval(EvolutionState state, int threadnum, Float[] inputs,
		 VectorIndividualCGP ind) {
		Float[] outputs = Evaluator.evaluate(state, threadnum, inputs, ind);
		return outputs;
	}
	
	protected void evalGPU(EvolutionState state, VectorIndividualCGP ind, NPRWithCGPOnGPUsProblem problem, int inputsNum, CudaData data, boolean isBig) {
			Evaluator.evaluateCUDAPopulation(state, ind, problem, inputsNum, data, isBig);
		}

}
