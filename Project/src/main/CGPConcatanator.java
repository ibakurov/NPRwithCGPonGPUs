package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Scanner;

public class CGPConcatanator {
	// Number of generations to analyze 
	static int howManyGenerationsShouldBeAnalyzed = 60;
	static int howManyIndividualsPerGeneration = 500;
	static int howManyObjectivesPerIndividual = 5;
	static int averageStatisticsOnHowManyGenerations = 5;
	
	/*
	 * Method that concatenate statistics files produced by the CGP algorithm 
	 */
	public static void averageStatisticsAtPath(String path) throws IOException {
		
		File file = new File(path);
		
		//Creating new file
		File fileWithAverageGenStats = new File("averageGen.stat");
		File fileWithAverageStats = new File("average.stat");
		File fileWithNormalizedStats = new File("normalized.stat");
		//If such file already exists
		if (!fileWithAverageGenStats.createNewFile()) {
			//Remove everything inside
			new PrintWriter(fileWithAverageGenStats.getPath()).close();
			System.out.println("averageGen.stat file already existed in the folder, I cleaned it.");
		} else {
			System.out.println("averageGen.stat file have been created.");
		}
		//If such file already exists
		if (!fileWithAverageStats.createNewFile()) {
			//Remove everything inside
			new PrintWriter(fileWithAverageStats.getPath()).close();
			System.out.println("average.stat file already existed in the folder, I cleaned it.");
		} else {
			System.out.println("average.stat file have been created.");
		}
		//If such file already exists
		if (!fileWithNormalizedStats.createNewFile()) {
			//Remove everything inside
			new PrintWriter(fileWithNormalizedStats.getPath()).close();
			System.out.println("normalized.stat file already existed in the folder, I cleaned it.");
		} else {
			System.out.println("normalized.stat file have been created.");
		}
		
		//Init array which will hold all the values
		ArrayList<ArrayList<ArrayList<Double>>> fitnessValuesForAllGensAndAllIndivs = new ArrayList<ArrayList<ArrayList<Double>>>();
		for (int i = 0; i < howManyGenerationsShouldBeAnalyzed; i++) {
			fitnessValuesForAllGensAndAllIndivs.add(new ArrayList<ArrayList<Double>>());
			for (int j = 0; j < howManyIndividualsPerGeneration; j++) {
				fitnessValuesForAllGensAndAllIndivs.get(i).add(new ArrayList<Double>());
			}
		}
		//Scanner to read from file
        Scanner scan;
		try {
            scan = new Scanner(file);
            for (int i = 0; i < howManyGenerationsShouldBeAnalyzed; i++) {
            	for (int j = 0; j < howManyIndividualsPerGeneration; j++) {
		        	if (scan.hasNextLine()) {
		        		for (int k = 0; k < howManyObjectivesPerIndividual; k++) {
		        			if (scan.hasNextDouble()) {
		        				fitnessValuesForAllGensAndAllIndivs.get(i).get(j).add(scan.nextDouble());
		        			} else {
		        				System.out.println("generation: " + i + "; individual: " + j + " doesn't have enough fitness scores for all objectvies");
		        			}
		        		}
		        	} else {
		        		System.out.println("file doesn't have enough data according to values");
		        	}
            	}
            }
        } catch (FileNotFoundException e1) {
        	e1.printStackTrace();
        }
		
		//Calculate all averages for all generation
		ArrayList<ArrayList<Double>> averageForGeneration = new ArrayList<ArrayList<Double>>();
		
		for (int i = 0; i < howManyGenerationsShouldBeAnalyzed; i++) {
			ArrayList<Double> sum = new ArrayList<Double>(howManyObjectivesPerIndividual);
			for (int k = 0; k < howManyObjectivesPerIndividual; k++) {
				sum.add(0.0);
			}
			for (int j = 0; j < howManyIndividualsPerGeneration; j++) {
				for (int k = 0; k < howManyObjectivesPerIndividual; k++) {
					sum.set(k, sum.get(k) + fitnessValuesForAllGensAndAllIndivs.get(i).get(j).get(k)/howManyIndividualsPerGeneration);
				}
			}
			averageForGeneration.add(sum);
		}
		
		//Calculate averages for every n generation
		ArrayList<ArrayList<Double>> averageOfAverages = new ArrayList<ArrayList<Double>>();
		
		for (int i = 0; i < howManyGenerationsShouldBeAnalyzed/averageStatisticsOnHowManyGenerations; i++) {
			ArrayList<Double> sum = new ArrayList<Double>(howManyObjectivesPerIndividual);
			for (int k = 0; k < howManyObjectivesPerIndividual; k++) {
				sum.add(0.0);
			}
			for (int j = i * averageStatisticsOnHowManyGenerations; j < (i + 1) * averageStatisticsOnHowManyGenerations; j++) {
				for (int k = 0; k < howManyObjectivesPerIndividual; k++) {
					sum.set(k, sum.get(k) + averageForGeneration.get(j).get(k)/averageStatisticsOnHowManyGenerations);
				}
			}
			averageOfAverages.add(sum);
		}
		
		//Normalize scores
		ArrayList<ArrayList<Double>> normalizedAverages = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> maxes = new ArrayList<Double>(howManyObjectivesPerIndividual);
		for (int k = 0; k < howManyObjectivesPerIndividual; k++) {
			maxes.add(0.0);
		}
		
		for (int i = 0; i < averageOfAverages.size(); i++) {
			for (int j = 0; j < averageOfAverages.get(i).size(); j++) {
				if (maxes.get(j) < averageOfAverages.get(i).get(j)) {
					maxes.set(j, averageOfAverages.get(i).get(j));
				}
			}
		}
		
		for (int i = 0; i < averageOfAverages.size(); i++) {
			normalizedAverages.add(new ArrayList<Double>());
			for (int j = 0; j < averageOfAverages.get(i).size(); j++) {
				normalizedAverages.get(i).add(averageOfAverages.get(i).get(j) / maxes.get(j));
			}
		}
		
		
		//Array to collect all data which should be written to the file
		ArrayList<String> linesWithDataForGenAverage = new ArrayList<String>();
		ArrayList<String> linesWithDataForAverage = new ArrayList<String>();
		ArrayList<String> linesWithDataForNormalized = new ArrayList<String>();

		for (int i = 0; i < averageForGeneration.size(); i++) {
			String line = "";
			for (int j = 0; j < averageForGeneration.get(i).size(); j++) {
				line += averageForGeneration.get(i).get(j) + " ";
			}
			linesWithDataForGenAverage.add(line);
		}
		
		for (int i = 0; i < averageOfAverages.size(); i++) {
			String line = "";
			for (int j = 0; j < averageOfAverages.get(i).size(); j++) {
				line += averageOfAverages.get(i).get(j) + " ";
			}
			linesWithDataForAverage.add(line);
		}
		
		for (int i = 0; i < normalizedAverages.size(); i++) {
			String line = "";
			for (int j = 0; j < normalizedAverages.get(i).size(); j++) {
				line += normalizedAverages.get(i).get(j) + " ";
			}
			linesWithDataForNormalized.add(line);
		}
		
		Path pathToWriteFileWithAverageGenData = Paths.get(fileWithAverageGenStats.getPath());
		Files.write(pathToWriteFileWithAverageGenData, linesWithDataForGenAverage, Charset.forName("UTF-8"));
		
		Path pathToWriteFileWithAverageData = Paths.get(fileWithAverageStats.getPath());
		Files.write(pathToWriteFileWithAverageData, linesWithDataForAverage, Charset.forName("UTF-8"));
		
		Path pathToWriteFileWithNormalizedData = Paths.get(fileWithNormalizedStats.getPath());
		Files.write(pathToWriteFileWithNormalizedData, linesWithDataForNormalized, Charset.forName("UTF-8"));
		
		System.out.println("DONE.");
	}
	
    public static void main(String[] args) {
    	try {
			CGPConcatanator.averageStatisticsAtPath("file.txt");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
