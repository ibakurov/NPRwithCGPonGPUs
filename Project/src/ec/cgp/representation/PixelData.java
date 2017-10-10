package ec.cgp.representation;

public class PixelData {
	// Coordinates
	public int x;
	public int y;
	
	// RGB values
	public double r;
	public double g;
	public double b;
	
	
	/**
	 * Constructors
	 */
	
	public PixelData (int x, int y, double red, double green, double blue) {
		this.x = x;
		this.y = y;
		this.r = red;
		this.g = green;
		this.b = blue;
	}

	public PixelData(int x, int y) {
		this.x = x;
		this.y = y;
	}
	
	public PixelData(double red, double green, double blue) {
		this.r = red;
		this.g = green;
		this.b = blue;
	}
}
