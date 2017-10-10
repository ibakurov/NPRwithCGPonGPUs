package main;

import com.sir_m2x.transscale.pointers.CudaFloat;
import com.sir_m2x.transscale.pointers.CudaFloat2D;
import com.sir_m2x.transscale.pointers.CudaInteger2D;

import ec.gp.cuda.CudaData;
import jcuda.Pointer;
import jcuda.driver.CUmodule;

/**
 * Record to represent a classification instance.
 */
public class MainRecord extends CudaData {

	private static final long serialVersionUID = -714391258208193581L;

	/** The size of the problem (i.e. the number of pixels we will send to GPU in one batch) */
	protected int problemSize = -1;
	
	public int[] x = null;
	public int[] y = null;
	public float[] redInput = null;
	public float[] greenInput = null;
	public float[] blueInput = null;
	public float[] redCanvas = null;
	public float[] greenCanvas = null;
	public float[] blueCanvas = null;
	public float[] opacity = null;
	public int[] expression1 = null;
	public int[] expression2 = null;
	public int[] expression3 = null;
	public float[] luminance = null;
	public float[] mean5x5 = null;
	public float[] std5x5 = null;
	public float[] min5x5 = null;
	public float[] max5x5 = null;
	public float[] mean7x7 = null;
	public float[] std7x7 = null;
    public float[] min7x7 = null;
    public float[] max7x7 = null;
    public float[] mean9x9 = null;
    public float[] std9x9 = null;
    public float[] min9x9 = null;
    public float[] max9x9 = null;
    public float[] mean11x11 = null;
    public float[] std11x11 = null;
    public float[] min11x11 = null;
    public float[] max11x11 = null;
    public float[] mean13x13 = null;
    public float[] std13x13 = null;
    public float[] min13x13 = null;
    public float[] max13x13 = null;
    public float[] ERC = null;
	
	public CudaInteger2D devX = null;
	public CudaInteger2D devY = null;
	public CudaFloat2D devRedInput = null;
	public CudaFloat2D devGreenInput = null;
	public CudaFloat2D devBlueInput = null;
	public CudaFloat2D devRedCanvas = null;
	public CudaFloat2D devGreenCanvas = null;
	public CudaFloat2D devBlueCanvas = null;
	public CudaFloat2D devOpacity = null;
	public CudaInteger2D devExpressions1 = null;
	public CudaInteger2D devExpressions2 = null;
	public CudaInteger2D devExpressions3 = null;
	public CudaFloat2D devLuminance = null;
	public CudaFloat2D devERC = null;
	public CudaFloat2D devmean5x5 = null;
	public CudaFloat2D devstd5x5 = null;
	public CudaFloat2D devmin5x5 = null;
	public CudaFloat2D devmax5x5 = null;
	public CudaFloat2D devmean7x7 = null;
	public CudaFloat2D devstd7x7 = null;
	public CudaFloat2D devmin7x7 = null;
	public CudaFloat2D devmax7x7 = null;
	public CudaFloat2D devmean9x9 = null;
	public CudaFloat2D devstd9x9 = null;
	public CudaFloat2D devmin9x9 = null;
	public CudaFloat2D devmax9x9 = null;
	public CudaFloat2D devmean11x11 = null;
	public CudaFloat2D devstd11x11 = null;
	public CudaFloat2D devmin11x11 = null;
	public CudaFloat2D devmax11x11 = null;
	public CudaFloat2D devmean13x13 = null;
	public CudaFloat2D devstd13x13 = null;
	public CudaFloat2D devmin13x13 = null;
	public CudaFloat2D devmax13x13 = null;
	
	//Did we already picked this pixel to apply brush stroke with it.
	public boolean haveBeenRandomned = false;

	public void init(int[] x, int[] y, float[] redInput, float greenInput[], float[] blueInput,
			float[] redCanvas, float[] greenCanvas, float[] blueCanvas, float[] opacity,
			float[] luminance, float[] ERC,
			 float[] mean5x5, float[] std5x5, float[] min5x5, float[] max5x5,
			 float[] mean7x7, float[] std7x7, float[] min7x7, float[] max7x7,
			 float[] mean9x9, float[] std9x9, float[] min9x9, float[] max9x9,
			 float[] mean11x11, float[] std11x11, float[] min11x11, float[] max11x11,
			 float[] mean13x13, float[] std13x13, float[] min13x13, float[] max13x13) {
		this.x = x;
		this.y = y;
		this.redInput = redInput;
		this.greenInput = greenInput;
		this.blueInput = blueInput;
		this.redCanvas = redCanvas;
		this.greenCanvas = greenCanvas;
		this.blueCanvas = blueCanvas;
		this.opacity = opacity;
		this.luminance = luminance;
		this.ERC = ERC;
		this.mean5x5 = mean5x5;
		this.std5x5 = std5x5;
		this.min5x5 = min5x5;
		this.max5x5 = max5x5;
		this.mean7x7 = mean7x7;
		this.std7x7 = std7x7;
		this.min7x7 = min7x7;
		this.max7x7 = max7x7;
		this.mean9x9 = mean9x9;
		this.std9x9 = std9x9;
		this.min9x9 = min9x9;
		this.max9x9 = max9x9;
		this.mean11x11 = mean11x11;
		this.std11x11 = std11x11;
		this.min11x11 = min11x11;
		this.max11x11 = max11x11;
		this.mean13x13 = mean13x13;
		this.std13x13 = std13x13;
		this.min13x13 = min13x13;
		this.max13x13 = max13x13;
		
		/**
		 * Now initialize 2D arrays with the height of 1 (which would in fact be 1D arrays)
		 * These will hold the training instances on the GPU
		 * Lazy transfer will tell TransScale not to immediately allocated the arrays on the device
		 * This is necessary as we do not know at this point which GPU will work on these arrays
		 * (In a multi-gpu setup, this is a very important consideration)
		 */
//		this.devExpectedOutput = new CudaDouble2D(problemSize, 1, 1, this.expectedOutput, true);
		
		this.devX = new CudaInteger2D(problemSize, 1, 1, this.x, true);
		this.devY = new CudaInteger2D(problemSize, 1, 1, this.y, true);
		this.devRedInput = new CudaFloat2D(problemSize, 1, 1, this.redInput, true);
		this.devGreenInput = new CudaFloat2D(problemSize, 1, 1, this.greenInput, true);
		this.devBlueInput = new CudaFloat2D(problemSize, 1, 1, this.blueInput, true);
		this.devRedCanvas = new CudaFloat2D(problemSize, 1, 1, this.redCanvas, true);
		this.devGreenCanvas = new CudaFloat2D(problemSize, 1, 1, this.greenCanvas, true);
		this.devBlueCanvas = new CudaFloat2D(problemSize, 1, 1, this.blueCanvas, true);
		this.devOpacity = new CudaFloat2D(problemSize, 1, 1, this.opacity, true);
		this.devLuminance = new CudaFloat2D(problemSize, 1, 1, this.luminance, true);
		this.devERC = new CudaFloat2D(problemSize, 1, 1, this.ERC, true);
		this.devmean5x5 = new CudaFloat2D(problemSize, 1, 1, this.mean5x5, true);
		this.devstd5x5 = new CudaFloat2D(problemSize, 1, 1, this.std5x5, true);
		this.devmin5x5 = new CudaFloat2D(problemSize, 1, 1, this.min5x5, true);
		this.devmax5x5 = new CudaFloat2D(problemSize, 1, 1, this.max5x5, true);
		this.devmean7x7 = new CudaFloat2D(problemSize, 1, 1, this.mean7x7, true);
		this.devstd7x7 = new CudaFloat2D(problemSize, 1, 1, this.std7x7, true);
		this.devmin7x7 = new CudaFloat2D(problemSize, 1, 1, this.min7x7, true);
		this.devmax7x7 = new CudaFloat2D(problemSize, 1, 1, this.max7x7, true);
		this.devmean9x9 = new CudaFloat2D(problemSize, 1, 1, this.mean9x9, true);
		this.devstd9x9 = new CudaFloat2D(problemSize, 1, 1, this.std9x9, true);
		this.devmin9x9 = new CudaFloat2D(problemSize, 1, 1, this.min9x9, true);
		this.devmax9x9 = new CudaFloat2D(problemSize, 1, 1, this.max9x9, true);
		this.devmean11x11 = new CudaFloat2D(problemSize, 1, 1, this.mean11x11, true);
		this.devstd11x11 = new CudaFloat2D(problemSize, 1, 1, this.std11x11, true);
		this.devmin11x11 = new CudaFloat2D(problemSize, 1, 1, this.min11x11, true);
		this.devmax11x11 = new CudaFloat2D(problemSize, 1, 1, this.max11x11, true);
		this.devmean13x13 = new CudaFloat2D(problemSize, 1, 1, this.mean13x13, true);
		this.devstd13x13 = new CudaFloat2D(problemSize, 1, 1, this.std13x13, true);
		this.devmin13x13 = new CudaFloat2D(problemSize, 1, 1, this.min13x13, true);
		this.devmax13x13 = new CudaFloat2D(problemSize, 1, 1, this.max13x13, true);
		
	}
	
	public void setExpressions(int[] expression1, int[] expression2, int[] expression3) {
		this.expression1 = expression1;
		this.expression2 = expression2;
		this.expression3 = expression3;
		
		this.devExpressions1 = new CudaInteger2D(expression1.length, 1, 1, this.expression1, true);
		this.devExpressions2 = new CudaInteger2D(expression2.length, 1, 1, this.expression2, true);
		this.devExpressions3 = new CudaInteger2D(expression3.length, 1, 1, this.expression3, true);
	}

	@Override
	public Pointer[] getArgumentPointers() {
		Pointer[] result = new Pointer[34];
		
		result[0] = devX.toPointer();
		result[1] = devY.toPointer();
		result[2] = devRedInput.toPointer();
		result[3] = devGreenInput.toPointer();
		result[4] = devBlueInput.toPointer();
		result[5] = devRedCanvas.toPointer();
		result[6] = devGreenCanvas.toPointer();
		result[7] = devBlueCanvas.toPointer();
		result[8] = devOpacity.toPointer();
		result[9] = devExpressions1.toPointer();
		result[10] = devExpressions2.toPointer();
		result[11] = devExpressions3.toPointer();
		result[12] = devLuminance.toPointer();
		result[13] = devERC.toPointer();
		result[14] = devmean5x5.toPointer();
		result[15] = devmean7x7.toPointer();
		result[16] = devmean9x9.toPointer();
		result[17] = devmean11x11.toPointer();
		result[18] = devmean13x13.toPointer();
		result[19] = devstd5x5.toPointer();
		result[20] = devstd7x7.toPointer();
		result[21] = devstd9x9.toPointer();
		result[22] = devstd11x11.toPointer();
		result[23] = devstd13x13.toPointer();
		result[24] = devmin5x5.toPointer();
		result[25] = devmin7x7.toPointer();
		result[26] = devmin9x9.toPointer();
		result[27] = devmin11x11.toPointer();
		result[28] = devmin13x13.toPointer();
		result[29] = devmax5x5.toPointer();
		result[30] = devmax7x7.toPointer();
		result[31] = devmax9x9.toPointer();
		result[32] = devmax11x11.toPointer();
		result[33] = devmax13x13.toPointer();
		
		return result;
	}

	@Override
	public long[] getKernelInputPitchInElements() {
//		long[] pitchDevX = this.devX.getDevPitchInElements();
//		long[] pitchDevY = this.devY.getDevPitchInElements();
//		long[] pitchDevRedInput = this.devRedInput.getDevPitchInElements();
//		long[] pitchDevGreenInput = this.devGreenInput.getDevPitchInElements();
//		long[] pitchDevBlueInput = this.devBlueInput.getDevPitchInElements();
//		long[] finalLong = new long[pitchDevX.length + pitchDevY.length + pitchDevRedInput.length + pitchDevGreenInput.length + pitchDevBlueInput.length];
//		for (int i = 0; i < finalLong.length; i++) {
//			if (i < pitchDevX.length) {
//				finalLong[i] = pitchDevX[i];
//			} else if (i < pitchDevX.length + pitchDevY.length) {
//				finalLong[i] = pitchDevY[i - pitchDevX.length];
//			} else if (i < pitchDevX.length + pitchDevY.length + pitchDevRedInput.length) {
//				finalLong[i] = pitchDevRedInput[i - pitchDevX.length - pitchDevY.length];
//			} else if (i < pitchDevX.length + pitchDevY.length + pitchDevRedInput.length + pitchDevGreenInput.length) {
//				finalLong[i] = pitchDevY[i - pitchDevX.length - pitchDevY.length - pitchDevRedInput.length];
//			} else if (i < pitchDevX.length + pitchDevY.length + pitchDevRedInput.length + pitchDevGreenInput.length + pitchDevBlueInput.length) {
//				finalLong[i] = pitchDevY[i - pitchDevX.length - pitchDevY.length - pitchDevRedInput.length - pitchDevGreenInput.length];
//			}
//		}
//		return finalLong;
		return this.devX.getDevPitchInElements();
	}

	@Override
	public void preInvocationTasks(CUmodule module) {
		devX.reallocate();
		devY.reallocate();
		devRedInput.reallocate();
		devGreenInput.reallocate();
		devBlueInput.reallocate();
		devRedCanvas.reallocate();
		devGreenCanvas.reallocate();
		devBlueCanvas.reallocate();
		devOpacity.reallocate();
		devExpressions1.reallocate();
		devExpressions2.reallocate();
		devExpressions3.reallocate();
		devLuminance.reallocate();
		devERC.reallocate();
		devmean5x5.reallocate();
		devmean7x7.reallocate();
		devmean9x9.reallocate();
		devmean11x11.reallocate();
		devmean13x13.reallocate();
		devstd5x5.reallocate();
		devstd7x7.reallocate();
		devstd9x9.reallocate();
		devstd11x11.reallocate();
		devstd13x13.reallocate();
		devmin5x5.reallocate();
		devmin7x7.reallocate();
		devmin9x9.reallocate();
		devmin11x11.reallocate();
		devmin13x13.reallocate();
		devmax5x5.reallocate();
		devmax7x7.reallocate();
		devmax9x9.reallocate();
		devmax11x11.reallocate();
		devmax13x13.reallocate();
	}

	@Override
	public void postInvocationTasks(CUmodule module) {
		devX.free();
		devY.free();
		devRedInput.free();
		devGreenInput.free();
		devBlueInput.free();
		devRedCanvas.free();
		devGreenCanvas.free();
		devBlueCanvas.free();
		devOpacity.free();
		devExpressions1.free();
		devExpressions2.free();
		devExpressions3.free();
		devLuminance.free();
		devERC.free();
		devmean5x5.free();
		devstd5x5.free();
		devmin5x5.free();
		devmax5x5.free();
		devmean7x7.free();
		devstd7x7.free();
		devmin7x7.free();
		devmax7x7.free();
		devmean9x9.free();
		devstd9x9.free();
		devmin9x9.free();
		devmax9x9.free();
		devmean11x11.free();
		devstd11x11.free();
		devmin11x11.free();
		devmax11x11.free();
		devmean13x13.free();
		devstd13x13.free();
		devmin13x13.free();
		devmax13x13.free();
	}
}