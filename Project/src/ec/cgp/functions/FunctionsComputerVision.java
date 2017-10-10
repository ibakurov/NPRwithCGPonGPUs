package ec.cgp.functions;

/**
 * Function set for the Breast Cancer (Wisconsin) classification problem.
 * 
 * @author David Oranchak, doranchak@gmail.com, http://oranchak.com
 * 
 */
public class FunctionsComputerVision implements Functions {

//	/** ERC */
//	static float ERC = new Random().nextFloat();
	
	/** add */
	static int F_ADD = 0;
	/** subtract */
	static int F_SUB = 1;
	/** multiply */
	static int F_MUL = 2;
	/** safe divide; return 1 if divisor is 0. */
	static int F_DIV = 3;
	/** negotiate a value */
	static int F_NEG = 4;
	/** sin */
	static int F_SIN = 5;
	/** cos */
	static int F_COS = 6;
	/** if less than zero then... else... */
	static int F_IFLEZ = 7;
	/** abs */
	static int F_ABS = 8;
	/** round */
	static int F_ROUND = 9;
	/** average */
	static int F_AVG = 10;
	/** log */
	static int F_LOG = 11;
	/** exp */
	static int F_EXP = 12;
	/** min */
	static int F_MIN = 13;
	/** max */
	static int F_MAX = 14;
	/** Brightening */
	static int F_BRT = 15;
	/** Darkening */
	static int F_DRK = 16;
	/** Burning */
	static int F_BRN = 17;
	/** Dodging */
	static int F_DGN = 18;
	/** Normal blending */
	static int F_NBLD = 19;
	/** Difference blending */
	static int F_DBLD = 20;
	/** Overlay blending */
	static int F_OBLD = 21;


	/** Interpret the given function and apply it to the given inputs. */
	public Float callFunction(Float[] inputs, int function, int numFunctions) {
		float[] arg = new float[inputs.length];
		for (int i = 0; i < inputs.length; i++)
			if (!(inputs[i] instanceof Float)) {
				arg[i] = 0f;
			} else {
				arg[i] = (Float) inputs[i];
			}
		if (function == F_ADD) {
//			System.out.println("+: " + (arg[0] + arg[1]));
			return arg[0] + arg[1];
		} else if (function == F_SUB) {
//			System.out.println("-: " + (arg[0] - arg[1]));
			return arg[0] - arg[1];
		} else if (function == F_MUL) {
//			System.out.println("*: " + arg[0] * arg[1]);
			return arg[0] * arg[1];
		} else if (function == F_DIV) {
//			System.out.println("/: " + arg[1] + " " + arg[0] / arg[1]);
			if (arg[1] == 0)
				return 1f;
			return (float)(arg[0] / arg[1]);
		} else if (function == F_NEG) {
//			System.out.println("neg: " + arg[0] + " " + (0-arg[0]));
			return 0-arg[0];
		} else if ( function == F_SIN) {
//			System.out.println("sin:" + (float)Math.sin(arg[0]));
			return (float)Math.sin(arg[0]);
		} else if ( function == F_COS) { 
//			System.out.println("cos: " + (float)Math.cos(arg[0]));
			return (float)Math.cos(arg[0]);
		} else if (function == F_IFLEZ) {
			if (arg[0] <= arg[1]) {
//				System.out.println("iflez: " + arg[2]);
				return arg[2];
			} else {
//				System.out.println("iflez: " + arg[3]);
				return arg[3];
			}
		}  else if (function == F_ABS) {
//			System.out.println("abs: " + Math.abs(arg[0]));
			return Math.abs(arg[0]);
		} else if (function == F_ROUND) {
//			System.out.println("round: " + Math.round(arg[0]));
			return (float) Math.round(arg[0]);
		} else if (function == F_AVG) {
//			System.out.println("avg: " + ( (arg[0] + arg[1]) / 2));
			return (arg[0] + arg[1]) / 2;
		} else if (function == F_LOG) {
//			System.out.println("log: " + arg[0] + " " + ( Math.log(Math.abs(arg[0]))));
			if (arg[0] == 0) {
				return 0.f;
			} else {
				return (float) Math.log(Math.abs(arg[0]));
			}
		} else if (function == F_EXP) {
//			System.out.println("exp: " + (arg[0]%10) + " " + ( Math.exp(arg[0]%10)));
			return (float) Math.exp(arg[0]%10);
		} else if (function == F_MIN) {
//			System.out.println("min: " + (  Math.min(arg[0], arg[1])));
			return (float) Math.min(arg[0], arg[1]);
		} else if (function == F_MAX) {
//			System.out.println("max: " + (  Math.max(arg[0], arg[1])));
			return (float) Math.max(arg[0], arg[1]);
		} else if (function == F_BRT) {
			float value = arg[0];
			if (value < 0) { value = 0.0f; }
			if (value > 1) { value = 1.0f; }
			float factor = arg[1];
	        if (factor > 1)  factor = factor - (int)factor;
//			System.out.println("brt: " + ( (((value * 255.0f) * (1 - factor) / 255.0f + factor) * 255.0f) / 255.0f));
	        return (((value * 255.0f) * (1 - factor) / 255.0f + factor) * 255.0f) / 255.0f;
		} else if (function == F_DRK) {
			float value = arg[0];
			if (value < 0) { value = 0.0f; }
			if (value > 1) { value = 1.0f; }
			float factor = arg[1];
	        if (factor > 1)  factor = factor - (int)factor;
//			System.out.println("drk: " + ( (((value * 255.0f) * (1 - factor) / 255.0f) * 255.0f) / 255.0f));
	        return (((value * 255.0f) * (1 - factor) / 255.0f) * 255.0f) / 255.0f;
		} else if (function == F_BRN) {
			float value1 = arg[0];
			float value2 = arg[1];
			if (value1 < 0) { value1 = 0.0f; }
			if (value1 > 1) { value1 = 1.0f; }
			if (value2 < 0) { value2 = 0.0f; }
			if (value2 > 1) { value2 = 1.0f; }
			float opacity = arg[2];
			opacity = (Math.abs(opacity));
			if (opacity == 0) { opacity = 0.1f; }
            if (opacity > 0.1) { opacity = 0.1f / opacity; }
            value1 = value1 * 255;
            value2 = value2 * 255;
    	    if (value2 == 0) { value2 = 1; }
    	    float finalValue = 255 - (255-value1)*(1 + 254*(255/value2)/255);
    	    finalValue = (value1 * (1 - opacity) + finalValue * opacity);
        	if (finalValue>255)	{ finalValue = 255; }
	        return finalValue / 255.0f;
		} else if (function == F_DGN) {
			float value1 = arg[0];
			float value2 = arg[1];
			if (value1 < 0) { value1 = 0.0f; }
			if (value1 > 1) { value1 = 1.0f; }
			if (value2 < 0) { value2 = 0.0f; }
			if (value2 > 1) { value2 = 1.0f; }
			float opacity = arg[2];
			opacity = (Math.abs(opacity));
			if (opacity == 0) { opacity = 0.1f; }
            if (opacity > 0.1) { opacity = 0.1f / opacity; }
            value1 = value1 * 255;
            value2 = value2 * 255;
    	    float finalValue = value1 * ( 1 + 254*value2 /255);
    	    finalValue = ( value1 * (1 - opacity) + finalValue * opacity);
        	if (finalValue>255)	{ finalValue = 255; }
	        return finalValue / 255.0f;
		} else if (function == F_NBLD) {
			float value1 = arg[0];
			float value2 = arg[1];
			if (value1 < 0) { value1 = 0.0f; }
			if (value1 > 1) { value1 = 1.0f; }
			if (value2 < 0) { value2 = 0.0f; }
			if (value2 > 1) { value2 = 1.0f; }
			float opacity = arg[2];
			opacity = (Math.abs(opacity));
			if (opacity == 0) { opacity = 0.1f; }
            if (opacity > 0.1) { opacity = 0.1f / opacity; }
            value1 = value1 * 255;
            value2 = value2 * 255;
    	    float finalValue = ( value1 * (1 - opacity) + value2 * opacity);
        	if (finalValue>255)	{ finalValue = 255; }
	        return finalValue / 255.0f;
		} else if (function == F_DBLD) {
			float value1 = arg[0];
			float value2 = arg[1];
			if (value1 < 0) { value1 = 0.0f; }
			if (value1 > 1) { value1 = 1.0f; }
			if (value2 < 0) { value2 = 0.0f; }
			if (value2 > 1) { value2 = 1.0f; }
			float opacity = arg[2];
			opacity = (Math.abs(opacity));
			if (opacity == 0) { opacity = 0.1f; }
            if (opacity > 0.1) { opacity = 0.1f / opacity; }
            value1 = value1 * 255;
            value2 = value2 * 255;
    	    float finalValue = ( (Math.abs(value1 - value2) * opacity)  + ( value2 * (1 - opacity)));
        	if (finalValue>255)	{ finalValue = 255; }
        	else if (finalValue<0)	{ finalValue = 0; }
	        return finalValue / 255.0f;
		} else if (function == F_OBLD) {
			float value1 = arg[0];
			float value2 = arg[1];
			if (value1 < 0) { value1 = 0.0f; }
			if (value1 > 1) { value1 = 1.0f; }
			if (value2 < 0) { value2 = 0.0f; }
			if (value2 > 1) { value2 = 1.0f; }
            value1 = value1 * 255;
            value2 = value2 * 255;
            float finalValue;
            if (value1 > 128) finalValue = 255 - (255- value1)*(255-value2)/128;
        	else finalValue = value1*value2 / 128;
        	if (finalValue>255)	{ finalValue = 255; }
        	else if (finalValue<0)	{ finalValue = 0; }
	        return finalValue / 255.0f;
		} else 
			throw new IllegalArgumentException("Function #" + function
					+ " is unknown.");
	}

	/**
	 * Interpret the given float as a boolean value. Any value > 0 is
	 * interpreted as "true".
	 */
	public static boolean f2b(float inp) {
		return inp > 0 ? true : false;
	}

	/** Convert the given boolean to float. "True" is 1.0; "false" is -1.0. */
	public static float b2f(boolean inp) {
		return inp ? 1f : -1f;
	}

	/**
	 * Return a function name, suitable for display in expressions, for the
	 * given function.
	 */
	public String functionName(int fn) {
		if (fn == F_ADD)
			return "+";
		if (fn == F_SUB)
			return "-";
		if (fn == F_MUL)
			return "*";
		if (fn == F_DIV)
			return "%";
		if (fn == F_NEG)
			return "neg";
		if (fn == F_SIN)
			return "sin";
		if (fn == F_COS)
			return "cos";
		if (fn == F_IFLEZ)
			return "iflez";
		if (fn == F_ABS)
			return "abs";
		if (fn == F_ROUND)
			return "round";
		if (fn == F_AVG)
			return "avg";
		if (fn == F_LOG)
			return "log";
		if (fn == F_EXP)
			return "exp";
		if (fn == F_MIN)
			return "min";
		if (fn == F_MAX)
			return "max";
		if (fn == F_BRT)
			return "brt";
		if (fn == F_DRK)
			return "drk";
		if (fn == F_BRN)
			return "brn";
		if (fn == F_DGN)
			return "dgn";
		if (fn == F_NBLD)
			return "nBld";
		if (fn == F_DBLD)
			return "dBld";
		if (fn == F_OBLD)
			return "oBld";
		else
			return "UNKNOWN FUNCTION";
	}

	/** Return the arity of the given function */
	public int arityOf(int fn) {
		if (fn == F_ADD)
			return 2;
		if (fn == F_SUB)
			return 2;
		if (fn == F_MUL)
			return 2;
		if (fn == F_DIV)
			return 2;
		if (fn == F_NEG)
			return 1;
		if (fn == F_SIN)
			return 1;
		if (fn == F_COS)
			return 1;
		if (fn == F_IFLEZ)
			return 4;
		if (fn == F_ABS)
			return 1;
		if (fn == F_ROUND)
			return 1;
		if (fn == F_AVG)
			return 2;
		if (fn == F_LOG)
			return 1;
		if (fn == F_EXP)
			return 1;
		if (fn == F_MIN)
			return 2;
		if (fn == F_MAX)
			return 2;
		if (fn == F_BRT)
			return 2;
		if (fn == F_DRK)
			return 2;
		if (fn == F_BRN)
			return 3;
		if (fn == F_DGN)
			return 3;
		if (fn == F_NBLD)
			return 3;
		if (fn == F_DBLD)
			return 3;
		if (fn == F_OBLD)
			return 2;
		else
			return 0;
	}

	/** Return the name, suitable for display, for the given input. */ 
	public String inputName(int inp) {
		if (inp == 0)
			return "X";
		if (inp == 1)
			return "Y";
		if (inp == 2)
			return "redInput";
		if (inp == 3)
			return "greenInput";
		if (inp == 4)
			return "blueInput";
		if (inp == 5)
			return "redCanvas";
		if (inp == 6)
			return "greenCanvas";
		if (inp == 7)
			return "blueCanvas";
		if (inp == 8)
			return "opacity";
		if (inp == 9)
			return "luminance";
		if (inp == 10)
			return "ERC";
		if (inp == 11)
			return "mean5x5";
		if (inp == 12)
			return "mean7x7";
		if (inp == 13)
			return "mean9x9";
		if (inp == 14)
			return "mean11x11";
		if (inp == 15)
			return "mean13x13";
		if (inp == 16)
			return "std5x5";
		if (inp == 17)
			return "std7x7";
		if (inp == 18)
			return "std9x9";
		if (inp == 19)
			return "std11x11";
		if (inp == 20)
			return "std13x13";
		if (inp == 21)
			return "min5x5";
		if (inp == 22)
			return "min7x7";
		if (inp == 23)
			return "min9x9";
		if (inp == 24)
			return "min11x11";
		if (inp == 25)
			return "min13x13";
		if (inp == 26)
			return "max5x5";
		if (inp == 27)
			return "max7x7";
		if (inp == 28)
			return "max9x9";
		if (inp == 29)
			return "max11x11";
		if (inp == 30)
			return "max13x13";
		else
			return "UNKNOWN INPUT";
	}
	
	
}
