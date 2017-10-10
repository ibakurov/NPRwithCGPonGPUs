package ec.cgp;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


import ec.EvolutionState;
import ec.cgp.functions.Functions;
import ec.cgp.representation.FloatVectorIndividual;
import ec.cgp.representation.VectorIndividualCGP;
import ec.cgp.representation.VectorSpeciesCGP;
import ec.gp.cuda.CudaData;
import ec.gp.cuda.CudaInterop;
import ec.gp.cuda.KernelOutputData;
import gnu.trove.list.TByteList;
import gnu.trove.list.array.TByteArrayList;
import main.MainRecord;
import main.NPRWithCGPOnGPUsProblem;

/**
 * Interprets the program encoded by an individual's genome, and evaluates the
 * results of the program with the given inputs.
 * 
 * @author David Oranchak, doranchak@gmail.com, http://oranchak.com
 * 
 */
public class Evaluator {

	static boolean DEBUG = false;

	/** counter to track number of node evaluations performed. */
	static int evals = 0;

	/**
	 * maps node evaluations to their node numbers. used to avoid re-processing
	 * nodes unnecessarily.
	 */
	public static List<Map<Integer, Float>> nodeMap;

	public static NPRWithCGPOnGPUsProblem p_problem;
	
	/**
	 * maps String representations of node sub-expressions to their node
	 * numbers. used to avoid re-processing nodes repeatedly.
	 */
	public static List<Map<Integer, String>> expressionMap;

	/** functions are loaded during setup. */
	public static Functions functions;
	
	/** Keeps the list for each thread for each output with expression list. */
	protected static List<List<TByteArrayList>> threadExpList;

	// checks to make sure that the Problem implements SimpleProblemForm
	public static void setup(final EvolutionState state, NPRWithCGPOnGPUsProblem p_problem) {
		Evaluator.p_problem = p_problem;
		// Initialize the list of list of expressions
		
		threadExpList = new ArrayList<List<TByteArrayList>>(state.evalthreads);
					
		for(int i = 0 ; i < state.evalthreads ; i++) {
			threadExpList.add(new ArrayList<TByteArrayList>());
		}
		
	}
	
	/**
	 * Evaluate the genome against the given inputs. If ind.expression is null,
	 * a string representation of the genome is computed and stored there.
	 * 
	 * @param inputs
	 *            inputs used to evaluate genome
	 * @param ind
	 *            the current individual
	 * 
	 * @return array of computed outputs from our Cartesian genetic program
	 */
	public static Float[] evaluate(EvolutionState state, int threadNum,
			Float[] inputs, VectorIndividualCGP ind) {
		nodeMap.get(threadNum).clear();
		expressionMap.get(threadNum).clear();
		
		VectorSpeciesCGP s = (VectorSpeciesCGP) ind.species;
		Float[] outputs = new Float[s.numOutputs];

		float[] gf = ((FloatVectorIndividual) ind).genome;
		

		boolean expression = false;
		StringBuffer sb = null;
		if (ind.expression == null) {
			expression = true;
			sb = new StringBuffer();
		}
		
		/** Evaluate results for each output node. */
		for (int i = 0; i < outputs.length; i++) {
			add(expression, sb, "o" + i + " = ");
			outputs[i] = evalNode(threadNum, expression, inputs, sb, ind
					.getGenome(), s.interpretFloat(gf.length - 1 - i,
					gf), s);
			
		}

		if (expression)
			ind.expression = sb;

		
		return outputs;
	}
	
	

	/**
	 * Computes the result of evaluating the given node.
	 * 
	 * @param threadNum
	 *            The current thread number.
	 * @param expression
	 *            If true, compute the string representation of the
	 *            sub-expression represented by this node.
	 * @param inputs The input values for this evaluation.
	 * 
	 * @param expr	Storage for the String-representation of the entire expression.
	 * @param genome The current genome.
	 * @param nodeNum The node number we are evaluating.
	 * @param s The CGP species
	 * @return The result of evaluation.
	 * 
	 * TODO: factor out all the float-vs-int checks to speed up evaluation.
	 * 
	 */
	private static Float evalNode(int threadNum, boolean expression,
			Float[] inputs, StringBuffer expr, Object genome, int nodeNum,
			VectorSpeciesCGP s) {
		Float val = nodeMap.get(threadNum).get(nodeNum);
		if (val != null) { /* We've already computed this node. */
			if (expression) /* append the already-computed expression string. */
				add(expression, expr, expressionMap.get(threadNum).get(nodeNum)); 
			return val; /* we already computed a result for this nodenumber, so
				just return the value. */
		}
		StringBuffer sb = null;
		if (expression)
			sb = new StringBuffer();
		if (nodeNum < s.numInputs) { // output may have hooked directly to an
			// input. check that here.
			nodeMap.get(threadNum).put(nodeNum, inputs[nodeNum]);
			if (expression) {
				add(expression, sb, ""
						+ functions.inputName(nodeNum));
				expressionMap.get(threadNum).put(nodeNum, sb.toString());
				expr.append(sb);
			}
			return inputs[nodeNum];
		}
		int pos = s.positionFromNodeNumber(nodeNum);
		int fn = (s.interpretFloat(pos, (float[]) genome));
		add(expression, sb, functions.functionName(fn));

		Float[] args = new Float[s.maxArity];
		for (int i = 0; i < functions.arityOf(fn); i++) { // eval each argument of the function

			int num = s.interpretFloat(pos + i + 1, (float[]) genome);
			if (num < s.numInputs) { // argument refers to an input (terminal) node.
				args[i] = inputs[num];
				add(expression, sb, " " + functions.inputName(num));
			} else { // argument refers to a function node.

				add(expression, sb, " (");
				args[i] = evalNode(threadNum, expression, inputs, sb, genome,
						num, s);
				add(expression, sb, ")");
			}
		}
		
		/* The arguments are ready now.  So, run the function. */
		Float result = functions.callFunction(args, fn, s.numFunctions);
		
		nodeMap.get(threadNum).put(nodeNum, result);
		if (expression) {
			expressionMap.get(threadNum).put(nodeNum, sb.toString());
			expr.append(sb);
		}
		return result;
	}

	/**
	 * Appends the given string to the given expression.
	 * @param expression If true, append; otherwise, ignore.
	 * @param sb The target expression.
	 * @param msg The snippet to append to the target.
	 */
	public static void add(boolean expression, StringBuffer sb, String msg) {
		if (expression)
			sb.append(msg);
	}

	public static void debug(String msg) {
		if (DEBUG)
			System.out.println(msg);
	}

	
	//---------------
	//CUDA
	//---------------
	
	/**
	 * A simple evaluator that doesn't do any coevolutionary evaluation.
	 * Basically it applies evaluation pipelines, one per thread, to various
	 * subchunks of a new population. Each thread is responsible for converting
	 * its own subchunk to a postfix expression.
	 */
	public static void evaluateCUDAPopulation(final EvolutionState state, VectorIndividualCGP ind, NPRWithCGPOnGPUsProblem problem, int inputsNum, CudaData data, boolean isBig) {

		int[] from = new int[state.evalthreads]; // starting index of this thread
		int[] to = new int[state.evalthreads];	// ending index of this thread
		
//		int offset = 0;
//		
		// These stuff should be done per subpopulation.
//		float[] genome = (float[])ind.getGenome();
//		for (float  : () {
//			CudaSubpopulation subPop = (CudaSubpopulation) sp;
//			// Determine the working scope of each thread
//			for (int i = 0 ; i < state.evalthreads ; i++) {
//				List<FloatVectorIndividual> listOfInd = subPop.needEval.get(i);
//				
//				from[i] = offset;
//				to[i] = from[i] + listOfInd.size() - 1;
//				offset += listOfInd.size();
//			}
			
			if (state.evalthreads == 1)
				traversePopChunk(state, (FloatVectorIndividual) ind, 0, inputsNum, data);
			else {
				Thread[] t = new Thread[state.evalthreads];
	
				// start up the threads
				for (int y = 0; y < state.evalthreads; y++) {
					ByteTraverseThread r = new ByteTraverseThread();
					r.threadnum = y;
					r.state = state;
					r.ind = ind;
					r.inputsNum = inputsNum;
					r.data = data;
					t[y] = new Thread(r);
					t[y].start();
				}
	
				// gather the threads
				for (int y = 0; y < state.evalthreads; y++)
					try {
						t[y].join();
					} catch (InterruptedException e) {
						state.output
								.fatal("Whoa! The main evaluation thread got interrupted!  Dying...");
					}
	
			}
			
			// Call the CUDA kernel using the defined problem data
			KernelOutputData[] outputs = CudaInterop.getInstance().evaluatePopulation(state, threadExpList, data);
			
			// call the assignFitness and assign fitnesses to each individual
			if (state.evalthreads == 1) {
				threadExpList.get(0).clear();
				applyRGB(state, ind, 0, outputs, from, to, problem, isBig);
			}
			else {
				Thread[] t = new Thread[state.evalthreads];
				
				// start up the threads
				for (int y = 0; y < state.evalthreads; y++) {
					// first clean this threads expressions
					
					threadExpList.get(y).clear();
					
					OutputAssignmentThread r = new OutputAssignmentThread();
					r.threadnum = y;
					r.state = state;
					r.ind = ind;
					r.outputs = outputs;
					r.from = from;
					r.to = to;
					r.problem = problem;
					r.isBig = isBig;
					t[y] = new Thread(r);
					t[y].start();
				}
	
				// gather the threads
				for (int y = 0; y < state.evalthreads; y++)
					try {
						t[y].join();
					} catch (InterruptedException e) {
						state.output
								.fatal("Whoa! The main evaluation thread got interrupted!  Dying...");
					}
			}
//		} // end-for (subpopulation)
		//Finished! :-)
	}

	protected static void traversePopChunk(EvolutionState state, FloatVectorIndividual ind, int threadnum, int inputsNum, CudaData data) {
		// Get the unevaluateds for the current thread
		List<TByteArrayList> myExpList = threadExpList.get(threadnum);
		float[] gf = ((FloatVectorIndividual) ind).genome;
		VectorSpeciesCGP s = (VectorSpeciesCGP) ind.species;
		Float[] outputs = new Float[s.numOutputs];
		StringBuffer sb = new StringBuffer();
		
		for (int i = 0; i < outputs.length; i++) {
			myExpList.add(new TByteArrayList());
		}
		
		/** Evaluate results for each output node. */
		//ATTENTION: We have 6 outputs, but first three have nothing to do with GPU, so we don't even calculate first three by i=3.
		for (int i = 3; i < outputs.length; i++) {
			// Add the terminating sequence! We will be reversing this guy later
			myExpList.get(i).add((byte)0);
			add(sb, "o" + i + " = ");
			postfixExpression(threadnum, inputsNum, sb, ind.getGenome(),
					s.interpretFloat(gf.length - 1 - i, gf), s, myExpList.get(i));
			myExpList.get(i).reverse();
		}
		
		//ATTENTION: We have 6 outputs, but first three have nothing to do with GPU, so we remove them from expList.
		myExpList.remove(0);
		myExpList.remove(0);
		myExpList.remove(0);
		
		ind.expression = sb;
		
		int[][] expressionsInt = new int[3][];
		for (int i = 0; i < myExpList.size(); i++) {
			expressionsInt[i] = new int[myExpList.get(i).size()];
			for (int j = 0; j < myExpList.get(i).size(); j++) {
				expressionsInt[i][j] = (int)myExpList.get(i).get(j);
			}
		}
		
		((MainRecord)data).setExpressions(expressionsInt[0], expressionsInt[1], expressionsInt[2]);
	}
	
	/**
	 * Computes the result of evaluating the given node.
	 * 
	 * @param threadNum
	 *            The current thread number.
	 * @param expression
	 *            If true, compute the string representation of the
	 *            sub-expression represented by this node.
	 * @param inputs The input values for this evaluation.
	 * 
	 * @param expr	Storage for the String-representation of the entire expression.
	 * @param genome The current genome.
	 * @param nodeNum The node number we are evaluating.
	 * @param s The CGP species
	 * @return The result of evaluation.
	 * 
	 * TODO: factor out all the float-vs-int checks to speed up evaluation.
	 * 
	 */
	private static void postfixExpression(int threadNum, int inputsNum, StringBuffer expr, Object genome, int nodeNum,
			VectorSpeciesCGP s, TByteList result) {

		StringBuffer sb = new StringBuffer();
		if (nodeNum < s.numInputs) { // output may have hooked directly to an
			// input. check that here.
			add(sb, "" + functions.inputName(nodeNum));
			expr.append(sb);
			TByteList newArray = new TByteArrayList();
			newArray.add((byte)(nodeNum+1));
			result.addAll(newArray);
			return;
		}
		
		int pos = s.positionFromNodeNumber(nodeNum);
		int fn = s.interpretFloat(pos, (float[]) genome);
		add(sb, functions.functionName(fn));

		TByteList newArray = new TByteArrayList();
		newArray.add((byte)(fn + inputsNum + 1));

		for (int i = 0; i < functions.arityOf(fn); i++) { // eval each argument of the function
			int num = s.interpretFloat(pos + i + 1, (float[]) genome);
			if (num < s.numInputs) { // argument refers to an input (terminal) node.
				newArray.add((byte)(num+1));
				add(sb, " " + functions.inputName(num));
			} else { // argument refers to a function node.
				add(sb, " (");
				postfixExpression(threadNum, inputsNum, sb, genome, num, s, newArray);
				add(sb, ")");
			}
		}
		
		result.addAll(newArray);
		expr.append(sb);
	}
	
	/**
	 * Appends the given string to the given expression.
	 * @param expression If true, append; otherwise, ignore.
	 * @param sb The target expression.
	 * @param msg The snippet to append to the target.
	 */
	public static void add(StringBuffer sb, String msg) {
		sb.append(msg);
	}
	
	/** A private helper class for implementing multithreaded byte traversal */
	private static class ByteTraverseThread implements Runnable {
		public EvolutionState state;
		public int threadnum;
		public VectorIndividualCGP ind;
		public int inputsNum;
		public CudaData data;
		
		public synchronized void run() {
			Evaluator.traversePopChunk(state, (FloatVectorIndividual) ind, threadnum, inputsNum, data);
		}
	}
	
	/** A private helper class for implementing multithreaded fitness assignment */
	private static class OutputAssignmentThread implements Runnable {
		public EvolutionState state;
		public int threadnum;
		public int[] from;
		public int[] to;
		public KernelOutputData[] outputs;
		public VectorIndividualCGP ind;
		public NPRWithCGPOnGPUsProblem problem;
		public boolean isBig;
		
		public synchronized void run() {
			Evaluator.applyRGB(state, ind, threadnum, outputs, from, to, problem, isBig);
		}
	}

	public static void applyRGB(EvolutionState state, VectorIndividualCGP ind, int threadnum, KernelOutputData[] outputs,
			int[] from, int[] to, NPRWithCGPOnGPUsProblem problem, boolean isBig) {
		
		for (int i = from[threadnum] ; i <= to[threadnum] ; i++) {
			
			// Ask the problem to assign a fitness value to this individual based
			// on the outputs of the kernel
			problem.applyRGB(state, ind, outputs[i], isBig);
			
			ind.evaluated = true; // Current individual is now evaluated :-)
		}
		
	}
	
}
