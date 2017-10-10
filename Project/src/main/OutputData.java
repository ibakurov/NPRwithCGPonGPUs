package main;

import ec.gp.cuda.KernelOutputData;

public class OutputData extends KernelOutputData {

	protected float[] output;
	
	@Override
	public void init(int count) {
		this.output = new float[count];
	}

	@Override
	public Object getUnderlyingData() {
		return this.output;
	}

	@Override
	public void setValueAt(int index, Object value) {
		this.output[index] = (float)value;
	}

	@Override
	public Object getValueAt(int index) {
		return this.output[index];
	}

	@Override
	public void copyValues(Object sourceArray, int start, int length) {
		float[] kernelOutput = (float[]) sourceArray;
		
		System.arraycopy(kernelOutput, start, output, 0, length);
	}

}
