/** =====================================Stack related definitions==================================== */
/** The size of the interpreter stack */
#define STACK_SIZE 128
#define push(A) do { sp++;stack[sp]=A; if(sp >= STACK_SIZE) printf("Stack overflow\n");} while(false)
#define pop(A) do{ A=stack[sp];sp--; if(sp < -1) printf("Stack underflow\n");} while(false)
/** ================================================================================================== */

/** The number of pixels, which we need to paint */
#define BLOCK_SIZE 512	// Used for the shared memory definitions

/************************************************************************************************************
 ************************************************************************************************************/

//TODO DOC: sadly there is only support for 1 pitch value for all input instances (which should be more than enough)
extern "C"
__global__ void evaluate(int* x, int* y,
	 												float* redInput, float* greenInput, float* blueInput,
													float* redCanvas, float* greenCanvas, float* blueCanvas,
													float* opacity,
	 												int* exp1, int* exp2, int* exp3,
													float* luminance, float* ERC,
													float* mean5x5, float* mean7x7, float* mean9x9, float* mean11x11, float* mean13x13,
													float* std5x5, float* std7x7, float* std9x9, float* std11x11, float* std13x13,
													float* min5x5, float* min7x7, float* min9x9, float* min11x11, float* min13x13,
													float* max5x5, float* max7x7, float* max9x9, float* max11x11, float* max13x13,
													int inputPitch, float* output, int outputPitch,
												/*const char** __restrict__ individuals,*/ const int pixelsPerBlockCount, const int pixelsCount/*, const int maxLength*/) {

	//In CUDA we have grids, which are blocks of threads!
	//Thus, blockIndex, goes from 0 to N-1 (Depends on what we instantiated it to be)
	//blockIdx.x is simply the index of the block in the grid
	int blockIndex = blockIdx.x;
	//threadIdx.x is the index of the thread inside the block
	int threadIndex = threadIdx.x;
	//blockDim.x is the number of threads per block
	int blockDimension = blockDim.x;

  //Check if this blockIndex is in the range of active pixels, we are working with
	if (blockIndex >= pixelsPerBlockCount)
		return;

	// Thread to data index mapping.
	int tid = blockIndex * blockDimension + threadIndex;

  //Check if our index is out of problem's size, thus there is not going to be data for this index.
	if (tid >= pixelsCount)
		return;

	//Also if this pixel is just white, we don't do anything with it
	if (opacity[tid] == 255) {
		output[(tid*3)] = -1;
		output[(tid*3) + 1] = -1;
		output[(tid*3) + 2] = -1;
		return;
	}

	float stack[STACK_SIZE];	// The stack is defined as the same type as the kernel output
	int sp;

		float rgb [3];

		for (int k = 0; k < 3; k++) {

				// Reset the stack pointer
				sp = - 1;

				//Get a proper expression to work with
				int* expression;
				if (k == 0) {
					expression = exp1;
				} else if (k == 1) {
					expression = exp2;
				} else {
					expression = exp3;
				}

				//This is used, in case if you want to comment out some of the inputs, so that we don't need to go and change all cases for functions.
				const int totalNumberOfTerminals = 31;

				int l = 0;	// Maintains the current index in the expression
				while(expression[l] != 0)
				{
					switch(expression[l])
					{
						case 1: {
							// printf("X (%i)", x[tid]);
							push(x[tid]);
						} break;
						case 2: {
							// printf("Y (%i)", x[tid]);
							push(y[tid]);
						} break;
						case 3: {
							// printf("redInput (%f)\n", redInput[tid]/ 255.f);
							push(redInput[tid]/ 255.f);
						} break;
						case 4: {
							// printf("greenInput (%i)", greenInput[tid]);
							push(greenInput[tid]/ 255.0);
						} break;
						case 5: {
							// printf("blueInput (%i)", blueInput[tid]);
							push(blueInput[tid]/ 255.0);
						} break;
						case 6: {
							// printf("redCanvas (%i)", redCanvas[tid]);
							push(redCanvas[tid]/ 255.0);
						} break;
						case 7: {
							// printf("greenCanvas (%i)", greenCanvas[tid]);
							push(greenCanvas[tid]/ 255.0);
						} break;
						case 8: {
							// printf("blueCanvas (%i)", blueCanvas[tid]);
							push(blueCanvas[tid]/ 255.0);
						} break;
						case 9: {
							// printf("opacity (%i)", opacity[tid]);
							push(opacity[tid]);
						} break;
						case 10: {
							// printf("luminance (%f)", luminance[tid]);
							push(luminance[tid]);
						} break;
						case 11: {
							// printf("ERC (%f)", ERC[tid]);
							push(ERC[tid]);
						} break;
						case 12: {
							// printf("mean5x5 (%f)", mean5x5[tid]);
							push(mean5x5[tid]);
						} break;
						case 13: {
							// printf("mean7x7 (%f)", mean7x7[tid]);
							push(mean7x7[tid]);
						} break;
						case 14: {
							// printf("mean9x9 (%f)", mean9x9[tid]);
							push(mean9x9[tid]);
						} break;
						case 15: {
							// printf("mean11x11 (%f)", mean11x11[tid]);
							push(mean11x11[tid]);
						} break;
						case 16: {
							// printf("mean13x13 (%f)", mean13x13[tid]);
							push(mean13x13[tid]);
						} break;
						case 17: {
							// printf("std5x5 (%f)", std5x5[tid]);
							push(std5x5[tid]);
						} break;
						case 18: {
							// printf("std7x7 (%f)", std7x7[tid]);
							push(std7x7[tid]);
						} break;
						case 19: {
							// printf("std9x9 (%f)", std9x9[tid]);
							push(std9x9[tid]);
						} break;
						case 20: {
							// printf("std11x11 (%f)", std11x11[tid]);
							push(std11x11[tid]);
						} break;
						case 21: {
							// printf("std13x13 (%f)", std13x13[tid]);
							push(std13x13[tid]);
						} break;
						case 22: {
							// printf("min5x5 (%f)", min5x5[tid]);
							push(min5x5[tid]);
						} break;
						case 23: {
							// printf("min7x7 (%f)", min7x7[tid]);
							push(min7x7[tid]);
						} break;
						case 24: {
							// printf("min9x9 (%f)", min9x9[tid]);
							push(min9x9[tid]);
						} break;
						case 25: {
							// printf("min11x11 (%f)", min11x11[tid]);
							push(min11x11[tid]);
						} break;
						case 26: {
							// printf("min13x13 (%f)", min13x13[tid]);
							push(min13x13[tid]);
						} break;
						case 27: {
							// printf("max5x5 (%f)", max5x5[tid]);
							push(max5x5[tid]);
						} break;
						case 28: {
							// printf("max7x7 (%f)", max7x7[tid]);
							push(max7x7[tid]);
						} break;
						case 29: {
							// printf("max9x9 (%f)", max9x9[tid]);
							push(max9x9[tid]);
						} break;
						case 30: {
							// printf("max11x11 (%f)", max11x11[tid]);
							push(max11x11[tid]);
						} break;
						case 31: {
							// printf("max13x13 (%f)", max13x13[tid]);
							push(max13x13[tid]);
						} break;
						case (totalNumberOfTerminals + 1): {
							// printf("+ ");
							float second;pop(second);
							float first;pop(first);
							float final = second + first;
							push(final);
						} break;
						case (totalNumberOfTerminals + 2): {
							// printf("- ");
							float second;pop(second);
							float first;pop(first);
							float final = second - first;
							push(final);
						} break;
						case (totalNumberOfTerminals + 3): {
							// printf("* ");
							float second;pop(second);
							float first;pop(first);
							float final = second * first;
							push(final);
						} break;
						case (totalNumberOfTerminals + 4): {
							// printf("/ ");
							float second;pop(second);
							float first;pop(first);
							if (second == 0) {
								float final = 1.0;
								push(final);
							} else {
								float final = second / first;
								push(final);
							}
						} break;
						case (totalNumberOfTerminals + 5): {
							// printf("neg ");
							float first;pop(first);
							float final = 0 - first;
							push(final);
						} break;
						case (totalNumberOfTerminals + 6): {
							// printf("sin ");
							float first;pop(first);
							float final = sinf(first);
							push(final);
						} break;
						case (totalNumberOfTerminals + 7): {
							// printf("cos ");
							float first;pop(first);
							float final = cosf(first);
							push(final);
						} break;
						case (totalNumberOfTerminals + 8): {
							// printf("iflez ");
							float fourth;pop(fourth);
							float third;pop(third);
							float second;pop(second);
							float first;pop(first);
							if (fourth <= third) {
								push(second);
							} else {
								push(first);
							}
						} break;
						case (totalNumberOfTerminals + 9): {
							// printf("abs ");
							float first;pop(first);
							float final = fabs(first);
							push(final);
						} break;
						case (totalNumberOfTerminals + 10): {
							// printf("round ");
							float first;pop(first);
							float final = round(first);
							push(final);
						} break;
						case (totalNumberOfTerminals + 11): {
							// printf("avg ");
							float second;pop(second);
							float first;pop(first);
							float final = (second + first) / 2;
							push(final);
						} break;
						case (totalNumberOfTerminals + 12): {
							// printf("log ");
							float first;pop(first);
							float final;
							if (first == 0) {
								final = 0.f;
							} else {
								final = log(fabs(first));
							}
							push(final);
						} break;
						case (totalNumberOfTerminals + 13): {
							// printf("exp ");
							float first;pop(first);
							float final = exp(fmodf(first, 10));
							push(final);
						} break;
						case (totalNumberOfTerminals + 14): {
							// printf("min ");
							float second;pop(second);
							float first;pop(first);
							float final = fminf(second, first);
							push(final);
						} break;
						case (totalNumberOfTerminals + 15): {
							// printf("max ");
							float second;pop(second);
							float first;pop(first);
							float final = fmaxf(second, first);
							push(final);
						} break;
						case (totalNumberOfTerminals + 16): {
							// printf("brt ");
							float value;pop(value);
							if (value < 0) { value = 0; }
							if (value > 1) { value = 1; }
							float factor;pop(factor);
							if (factor > 1) { factor = factor - (int)factor; }
							float final = (((value * 255.0) * (1 - factor) / 255.0 + factor) * 255.0) / 255.0;
							push(final);
						} break;
						case (totalNumberOfTerminals + 17): {
							// printf("drk ");
							float factor;pop(factor);
							float value;pop(value);
							if (value < 0) { value = 0; }
							if (value > 1) { value = 1; }
							if (factor > 1) { factor = factor - (int)factor; }
							float final = (((value * 255.0) * (1 - factor) / 255.0) * 255.0) / 255.0;
							push(final);
						} break;
						case (totalNumberOfTerminals + 18): {
							// printf("brn ");
							float opacityValue;pop(opacityValue);
							float value2;pop(value2);
							float value1;pop(value1);
							if (value1 < 0) { value1 = 0.f; }
							if (value1 > 1) { value1 = 1.f; }
							if (value2 < 0) { value2 = 0.0f; }
							if (value2 > 1) { value2 = 1.0f; }
							opacityValue = fabs(opacityValue);
							if (opacityValue == 0) { opacityValue = 0.1f; }
							if (opacityValue > 0.1) { opacityValue = 0.1f / opacityValue; }
							value1 = value1 * 255.0f;
							value2 = value2 * 255.0f;
							if (value2 == 0) { value2 = 1.0f; }
							float final = 255.0f - (255.0f-value1)*(1.0f + 254.0f*(255.0f/value2)/255.0f);
							final = (value1 * (1.0f - opacityValue) + final * opacityValue);
							if (final>255)	{ final = 255.0f; }
							final = final / 255.0f;
							push(final);
						} break;
						case (totalNumberOfTerminals + 19): {
							// printf("dgn ");
							float opacityValue;pop(opacityValue);
							float value2;pop(value2);
							float value1;pop(value1);
							if (value1 < 0) { value1 = 0; }
							if (value1 > 1) { value1 = 1; }
							if (value2 < 0) { value2 = 0; }
							if (value2 > 1) { value2 = 1; }
							opacityValue = fabs(opacityValue);
							if (opacityValue == 0) { opacityValue = 0.1; }
							if (opacityValue > 0.1) { opacityValue = 0.1 / opacityValue; }
							value1 = value1 * 255;
							value2 = value2 * 255;
							float final = ( value1 * (1 - opacityValue) + value2 * opacityValue);
							if (final>255)	{ final = 255; }
							final = final / 255.0;
							push(final);
						} break;
						case (totalNumberOfTerminals + 20): {
							// printf("nbld ");
							float opacityValue;pop(opacityValue);
							float value2;pop(value2);
							float value1;pop(value1);
							if (value1 < 0) { value1 = 0; }
							if (value1 > 1) { value1 = 1; }
							if (value2 < 0) { value2 = 0; }
							if (value2 > 1) { value2 = 1; }
							opacityValue = fabs(opacityValue);
							if (opacityValue == 0) { opacityValue = 0.1; }
							if (opacityValue > 0.1) { opacityValue = 0.1 / opacityValue; }
							value1 = value1 * 255;
							value2 = value2 * 255;
							float final = ( value1 * (1 - opacityValue) + value2 * opacityValue);
							if (final>255)	{ final = 255; }
							final = final / 255.0;
							push(final);
						} break;
						case (totalNumberOfTerminals + 21): {
							// printf("dbld ");
							float opacityValue;pop(opacityValue);
							float value2;pop(value2);
							float value1;pop(value1);
							if (value1 < 0) { value1 = 0; }
							if (value1 > 1) { value1 = 1; }
							if (value2 < 0) { value2 = 0; }
							if (value2 > 1) { value2 = 1; }
							opacityValue = fabs(opacityValue);
							if (opacityValue == 0) { opacityValue = 0.1; }
							if (opacityValue > 0.1) { opacityValue = 0.1 / opacityValue; }
							value1 = value1 * 255;
							value2 = value2 * 255;
							float final = ( (fabs(value1 - value2) * opacityValue)  + ( value2 * (1 - opacityValue)));
							if (final>255)	{ final = 255; }
							else if (final<0)	{ final = 0; }
							final = final / 255.0;
							push(final);
						} break;
						case (totalNumberOfTerminals + 22): {
							// printf("obld ");
							float value2;pop(value2);
							float value1;pop(value1);
							if (value1 < 0) { value1 = 0; }
							if (value1 > 1) { value1 = 1; }
							if (value2 < 0) { value2 = 0; }
							if (value2 > 1) { value2 = 1; }
							value1 = value1 * 255;
							value2 = value2 * 255;
							float final;
							if (value1 > 128) { final = 255 - (255- value1)*(255-value2)/128; }
        			else { final = value1*value2 / 128; }
        			if (final>255)	{ final = 255; }
        			else if (final<0)	{ final = 0; }
							final = final / 255.0;
							push(final);
						} break;
						default:printf("Unrecognized OPCODE in the expression tree!");break;
					}
					// printf("sp: %d", sp);
					l++;
				}

				if (l == 0) {
					printf("Expression is empty %i!\n", tid);
					rgb[k] = 255;
				} else {
						// Pop the top of the stack
						float stackTop;
						pop(stackTop);

						if(sp!=-1) {
							printf("Stack pointer is not -1 but is %d\n", sp);
						}

						rgb[k] = stackTop;
				}
		}

		// Assign the results to outputs
		output[(tid*3)] = rgb[0];
		output[(tid*3) + 1] = rgb[1];
		output[(tid*3) + 2] = rgb[2];
}
