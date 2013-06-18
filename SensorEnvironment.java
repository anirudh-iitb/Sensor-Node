import java.util.Random;
import java.util.Vector;
import java.lang.Math;
import org.rlcommunity.rlglue.codec.EnvironmentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.types.Reward_observation_terminal;
import org.rlcommunity.rlglue.codec.util.EnvironmentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpecVRLGLUE3;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;

public class SensorEnvironment implements EnvironmentInterface
{
	WorldDescription theWorld;
	double mean_x = 1.5;
	double mean_y = 10.0;

	public String env_init()
	{
		
		TaskSpecVRLGLUE3 theTaskSpecObject = new TaskSpecVRLGLUE3();
		theTaskSpecObject.setDiscountFactor(0.9d);
		theTaskSpecObject.addDiscreteObservation(new IntRange(0,2600));
		theTaskSpecObject.addDiscreteAction(new IntRange(0,50));
		theTaskSpecObject.setRewardRange(new DoubleRange(0,50));
		String taskSpecString = theTaskSpecObject.toTaskSpec();
		TaskSpec.checkTaskSpec(taskSpecString);
		return taskSpecString;
	}

	public Observation env_start()
	{
		theWorld = new WorldDescription(mean_x,mean_y);
		Observation observation = new Observation(1,0,0);
		observation.setInt(0,theWorld.getState());
		return observation;
	}

	public Reward_observation_terminal env_step(Action thisAction)
	{
		Reward_observation_terminal RewardObs = new Reward_observation_terminal();
		RewardObs.setReward(theWorld.getReward(thisAction.getInt(0)));
		theWorld.updateState(thisAction.getInt(0));
		Observation observation = new Observation(1,0,0);
		observation.setInt(0,theWorld.getState());
		RewardObs.setObservation(observation);
		RewardObs.setTerminal(theWorld.isTerminal());
		return RewardObs;
	}

	public void env_cleanup()
	{
		theWorld = null;
	}

	public String env_message(String message)
	{
		double x = Double.parseDouble(message);
		mean_x = x;
		return "none";
	}

	public static void main(String[] args)
	{
		EnvironmentLoader theLoader = new EnvironmentLoader(new SensorEnvironment());
        theLoader.run();
	}
}

class WorldDescription
{
	private int DMAX = 50;
	private int EMAX = 50;

	private Random randGenerator = new Random();

	//THE STATE VARIABLES Q AND E
	private int q_k;
	private int e_k;

	//THE RANDOM VARIABLES X AND Y
	private int x_k;
	private int y_k;

	//MEAN OF X AND Y
	private double lambda_x = 1.5;
	private double lambda_y = 10.0;

	//THE COST FUNCTION PARAMETERES r1 and r2
	private double r1 = 1.0;
	private double r2 = 0.0;

	public WorldDescription(double x, double y)
	{
		//q_k = randGenerator.nextInt(DMAX+1);
		//e_k = randGenerator.nextInt(EMAX+1);
		//SETTING LAMBDA_X AND LAMBDA_Y
		lambda_x = x;
		lambda_y = y;

		//GIVING FULL BUFFERS INITIALLY
		q_k = 0;
		e_k = 0;
	}

	public boolean isTerminal()
	{
		return false;
	}

	public int getState()
	{
		return q_k*51+e_k;
	}

	public int getNumStates()
	{
		return (DMAX+1)*(EMAX+1);
	}

	public double getReward(int action)
	{
		return (r1*Math.max((q_k - conversion(action)),0))+(r2*action);
	}

	public void updateState(int action)
	{	
		x_k = generate_x();
		y_k = generate_y();
		q_k = Math.max((q_k - conversion(action)),0) + x_k;
		e_k = e_k - action + y_k;
		if(q_k>DMAX) q_k = DMAX;
		if(e_k>EMAX) e_k = EMAX;
	}

	public int conversion(int t)//THE CONVERSION FUNCTION 'g' FROM ENERGY TO DATA
	{
		return (int)(1.0*(Math.log(1+t)/Math.log(2.73)));
	}

	public int generate_x()//THE FUNCTION TO GENERATE RANDOM VARIABLE X ACC TO POISSON DISTRIBUTION
	{
		double L = Math.exp(-lambda_x);
		double p = 1.0;
		int k = 0;

		do{
			k++;
			p *= Math.random();
		} while(p>L);

		return k-1;
	}

	public int generate_y()//THE FUNCTION TO GENERATE RANDOM VARIABLE Y ACC TO POISSON DISTRIBUTION
	{
		double L = Math.exp(-lambda_y);
		double p = 1.0;
		int k =0;

		do{
			k++;
			p *= Math.random();
		} while(p>L);

		return k-1;
	}
}

