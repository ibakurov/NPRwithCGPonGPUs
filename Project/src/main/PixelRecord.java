package main;
import ec.cgp.Record;

/**
 * Record to represent a classification instance.
 */
public class PixelRecord implements Record {

	public int x;
	public int y;
	public float redInput;
	public float greenInput;
	public float blueInput;
	public float redCanvas;
	public float greenCanvas;
	public float blueCanvas;
	public float opacity;

	//Did we already coloured this pixel.
	public boolean haveBeenColoured = false;

	public String toString() {
		return "opacity: " + opacity + ";\n" + "coordinates: {x: " + x + ", y: " + y + "};\n" 
				+ "canvas: {red: " + redCanvas + ", green: " + greenCanvas + ", blue: " + blueCanvas + "};\n" 
				+ "input: {red: " + redInput + ", green: " + greenInput + ", blue: " + blueInput + "};\n";
	}
	
	public void init(int x, int y, float redInput, float greenInput, float blueInput, float redCanvas, float greenCanvas, float blueCanvas, float opacity) {
		this.x = x;
		this.y = y;
		this.redInput = redInput;
		this.greenInput = greenInput;
		this.blueInput = blueInput;
		this.redCanvas = redCanvas;
		this.greenCanvas = greenCanvas;
		this.blueCanvas = blueCanvas;
		this.opacity = opacity;
	}
}
