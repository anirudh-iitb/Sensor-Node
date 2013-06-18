import java.lang.Math;
import java.util.Vector;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec; 

public class SensorAgent implements AgentInterface
{
	private Random randGenerator = new Random();
	private Action lastAction;
	private Observation lastObservation;
	private double[][] valuefunction = null;

	private int[] visitNumber = null;

	private double alpha_stepsize = 0.1;
	private double discount = 0.9;
	private double epsilon = 0.1;
	private double initial = 0.0;
	private double numSteps = 1.0;//NUMBER OF STEPS FOR DECAYING STEPSIZE
	private double average_reward = 0.0;
	private double beta_stepsize = 0.1;

	private int numStates;
	private int numActions;

	private boolean freezeLearning = false;
	private boolean startDecay = false;

	//************************************ CONTROL ALGORITHM ***********************************

	private boolean qlearning = false;
	private boolean rel_qlearning = false;
	private boolean greedy = false;
	private boolean mto = false;
	private boolean rlearning = false;

	double avg_reward;//AVERAGE COST WHILE EVALUATING THE AGENT
	double numOfSteps;//NUMBER OF STEPS FOR CALCULATING AVERAGE COST


	int fixedState = 86;
	//******************************************************************************************



	public void agent_init(String taskspec)
	{

		avg_reward = 0.0;
		numOfSteps = 0.0;
		numSteps = 1.0;
		alpha_stepsize = 0.1;
		average_reward = 0.0;
		beta_stepsize = 0.1;

		TaskSpec spec = new TaskSpec(taskspec);
		numStates = spec.getDiscreteObservationRange(0).getMax()+1;
		numActions = spec.getDiscreteActionRange(0).getMax()+1;

		valuefunction = new double[numActions][numStates];

		visitNumber = new int[numStates];

		for(int i=0; i<numStates; i++)
		{
			for(int j=0;j<numActions;j++)
			{
				valuefunction[j][i]=initial;
			}

			visitNumber[i] = 0;

		}

		discount = spec.getDiscountFactor();
	}

	public Action agent_start(Observation observation)
	{
		int newActionInt = egreedy(observation.getInt(0));

		visitNumber[observation.getInt(0)]++;

		Action returnAction = new Action(1,0,0);
		returnAction.intArray[0] = newActionInt;

		lastAction = returnAction;
		lastObservation = observation;

		return returnAction;
	}

	public Action agent_step(double reward, Observation observation)
	{
		//*********
		//System.out.println("Cost is "+ reward);
		//********
		int newStateInt = observation.getInt(0);
		int lastStateInt = lastObservation.getInt(0);
		int lastActionInt = lastAction.getInt(0);

		visitNumber[newStateInt]++;

		int newActionInt = egreedy(newStateInt);

		//assert (newActionInt<=newStateInt%51);
		if(newActionInt>newStateInt%51) System.out.println("FAIL");
		if(rlearning)
		{
			int minActionInt = 0;
			for(int i=0; i<numActions; i++)
			{
				if(valuefunction[i][newStateInt]<valuefunction[minActionInt][newStateInt]) minActionInt = i;
			}

			double Q_sa = valuefunction[lastActionInt][lastStateInt];
			double Q_sprime_aprime = valuefunction[minActionInt][newStateInt];

			double new_Q_sa = Q_sa + alpha_stepsize*(reward - average_reward + Q_sprime_aprime - Q_sa);
			if(!freezeLearning){
				valuefunction[lastActionInt][lastStateInt] = new_Q_sa;
				double minValue = new_Q_sa;
				for(int i=0;i<numActions; i++)
				{
					if(valuefunction[i][lastStateInt]<minValue) minValue = valuefunction[i][lastStateInt];
				}
				if(minValue==new_Q_sa)
				{
					average_reward = average_reward + beta_stepsize*(reward - average_reward + Q_sprime_aprime - minValue);
				}
				if(startDecay){
					alpha_stepsize = (alpha_stepsize * numSteps)/(numSteps+1);
					beta_stepsize = (beta_stepsize * numSteps)/(numSteps+1);
					numSteps++;
				}
			}
		}
		else if(qlearning)
		{
			int minActionInt=0;
			for(int i=0; i<numActions; i++)
			{
				if(valuefunction[i][newStateInt]<valuefunction[minActionInt][newStateInt]) minActionInt = i;
			}

			double Q_sa = valuefunction[lastActionInt][lastStateInt];
			double Q_sprime_aprime = valuefunction[minActionInt][newStateInt];

			double new_Q_sa = Q_sa + alpha_stepsize*(reward + discount*Q_sprime_aprime - Q_sa);

			if(!freezeLearning){
				valuefunction[lastActionInt][lastStateInt] = new_Q_sa;
				if(startDecay){ 
					alpha_stepsize = (alpha_stepsize * numSteps)/(numSteps+1);
					numSteps++;
				}
			}
		}
		else if(rel_qlearning)
        {
            int minActionInt = 0;
            for(int i=0; i<numActions; i++)
			{
				if(valuefunction[i][newStateInt]<valuefunction[minActionInt][newStateInt]) minActionInt = i;
			}

            int minFixedActionInt = 0;
            for(int i=0; i<numActions; i++)
			{
				if(valuefunction[i][fixedState]<valuefunction[minFixedActionInt][fixedState]) minFixedActionInt = i;
			}

            double Q_sa = valuefunction[lastActionInt][lastStateInt];
            double Q_sprime_aprime = valuefunction[minActionInt][newStateInt];

            double Q_sa_fixed = valuefunction[minFixedActionInt][fixedState];

            double new_Q_sa = Q_sa + alpha_stepsize*(reward + Q_sprime_aprime - Q_sa - Q_sa_fixed);

            if(!freezeLearning){
            	valuefunction[lastActionInt][lastStateInt] = new_Q_sa;
            	if(startDecay){
            		alpha_stepsize = (alpha_stepsize*numSteps)/(numSteps+1);
            		numSteps++;
            	}
            }

        }

		Action returnAction = new Action();
		returnAction.intArray = new int[]{newActionInt};

		lastAction = returnAction.duplicate();
		lastObservation = observation.duplicate();

		if(freezeLearning)
        {
            double new_reward = avg_reward + (1/(numOfSteps+1))*(reward - avg_reward);
            avg_reward = new_reward;
            numOfSteps++;
        }

		return returnAction;
	}

	public void agent_end(double reward)
	{
		int lastStateInt = lastObservation.getInt(0);
		int lastActionInt = lastAction.getInt(0);

		double Q_sa = valuefunction[lastActionInt][lastStateInt];
		if(qlearning)
		{
			double new_Q_sa = Q_sa + alpha_stepsize*(reward - Q_sa);
			if(!freezeLearning){
				valuefunction[lastActionInt][lastStateInt] = new_Q_sa;
				if(startDecay){
					alpha_stepsize = (alpha_stepsize*numSteps)/(numSteps+1);
					numSteps++;
				}
			}
		}

		else if(rel_qlearning)
        {
            int action=0;
            for(int i=0; i<numActions; i++)
            {
            	if(valuefunction[i][fixedState]<valuefunction[action][fixedState]) action =i;
            }
            double Q_sa_fixed = valuefunction[action][fixedState];
            double new_Q_sa = Q_sa + alpha_stepsize*(reward - Q_sa - Q_sa_fixed);

            if(!freezeLearning){
            	valuefunction[lastActionInt][lastStateInt] = new_Q_sa;
            	if(startDecay){
            		alpha_stepsize = (alpha_stepsize*numSteps)/(numSteps+1);
            		numSteps++;
            	}
            }
        }

		lastAction = null;
		lastObservation = null;

		if(freezeLearning)
        {
            avg_reward = avg_reward + (1/(numOfSteps+1))*(reward - avg_reward);
            numOfSteps++;
        }
	}

	public void agent_cleanup()
	{
		lastObservation = null;
		lastAction = null;
		valuefunction = null;
	}

	public String agent_message(String message)
	{
		if(message.equals("Freeze Learning"))
		{
			freezeLearning = true;
			return "Got it";
		}
		else if(message.equals("UnFreeze Learning"))
		{
			freezeLearning = false;
			return "Got it";
		}
		else if(message.equals("print-policy"))
		{
			print_policy();
			return "Got it";
		}
		else if(message.equals("print-average-cost"))
		{
			System.out.println("The average cost is " + avg_reward);
			return "Got it";
		}
		else if(message.equals("print-value-function"))
		{
			print_value_function();
			return "Got it";
		}
		else if(message.equals("print-maximum-visited-state"))
		{
			int max=0;
			for(int i=0;i<numStates;i++)
			{
				if(visitNumber[i]>visitNumber[max]) max = i;
			}
			System.out.println("Most visited state is "+ max);
			return "Got it";
		}
		else if(message.equals("start-decay"))
		{
			startDecay = true;
			return "Got it";
		}
		else if(message.equals("stop-decay"))
		{
			startDecay = false;
			return "Got it";
		}
		else if(message.equals("greedy-policy"))
		{
			greedy = true;
			return "Got it";
		}
		else if(message.equals("stop-greedy"))
		{
			greedy = false;
			return "Got it";
		}
		else if(message.equals("discounted-qlearning"))
		{
			qlearning = true;
			return "Got it";	
		}
		else if(message.equals("stop-discounted-qlearning"))
		{
			qlearning = false;
			return "Got it";
		}
		else if(message.equals("relative-qlearning"))
		{
			rel_qlearning = true;
			return "Got it";
		}
		else if(message.equals("stop-relative-qlearning"))
		{
			rel_qlearning = false;
			return "Got it";
		}
		else if(message.equals("mto"))
		{
			mto = true;
			return "Got it";
		}
		else if(message.equals("stop-mto"))
		{
			mto = false;
			return "got it";
		}
		else if(message.equals("rlearning"))
		{
			rlearning = true;
			return "Got it";
		}
		else if(message.equals("stop-rlearning"))
		{
			rlearning = false;
			return "Got it";
		}
		return "none";
	}

	public int egreedy(int state)
	{
		
		if(mto)
		{
			int energy = state%51;
			int data = state/51;

			double E_Y = Math.log(11)/Math.log(2.73);
			double value = 0.99*(E_Y + 0.001*(energy-0.1*data));
			if(value>=energy)
			{
				if(data > (int)(Math.log(1+energy)/Math.log(2.73))) return energy;
				else
				{
					double energy_needed = Math.exp(data/1.0) - 1;
					int energy_required = ((int) energy_needed)+1;
					int t = Math.min(energy,energy_required);
					return t;
				}
			}
			else
			{
				if(data > (int)(Math.log(1+value)/Math.log(2.73))) return (int)value;
				else
				{
					double energy_needed = Math.exp(data/1.0) - 1;
					int energy_required = ((int) energy_needed)+1;
					int t = Math.min((int)value,energy_required);
					return t;
				}
			}
		}
		else if(greedy)
		{
			int energy = state%51;
			int data = state/51;

			if(data > (int)(Math.log(1+energy)/Math.log(2.73))) return energy;
			else
			{
				//System.out.println("The state is "+ state);
				double energy_needed = Math.exp(data/1.0) - 1;
				int energy_required = ((int) energy_needed)+1;

				int t = Math.min(energy,energy_required);

				//System.out.println("The action is "+ t);
				return t;
			}
		}
		else{
			int maxValue = state%51;
			if(randGenerator.nextDouble()<epsilon)
			{
				return randGenerator.nextInt(maxValue+1);
			}
			else
			{
				int minActionInt = 0;
				for(int i=0; i<maxValue+1; i++)
				{
					if(valuefunction[i][state]<valuefunction[minActionInt][state]) minActionInt = i;
				}
				return minActionInt;
			}
		}
	}

	/*public int egreedy(int state)
	{
		int energy = state%51;
		int data = state/51;

		if(data > (int)(Math.log(1+energy)/Math.log(2.73))) return energy;
		else
		{
			//System.out.println("The state is "+ state);
			double energy_needed = Math.exp(data/1.0) - 1;
			int energy_required = ((int) energy_needed)+1;

			int t = Math.min(energy,energy_required);

			//System.out.println("The action is "+ t);
			return t;
		}
	}*/

	public static void main(String[] args)
	{
		AgentLoader theLoader = new AgentLoader(new SensorAgent());
        theLoader.run();
	}


	public void print_policy()
	{
		for(int s = 0; s<numStates;s++)
		{
			int data = s/51;
			int energy = s%51;
			int minIndex = 0;
			for(int i=0; i<=energy; i++)
			{
				if(valuefunction[i][s]<valuefunction[minIndex][s]) minIndex = i;
			}
			System.out.println("For the state "+ s+ " with data "+ data + " and energy "+ energy+ " the action is "+ minIndex);	
		}
	}

	public void print_value_function()
	{
		for(int s=0;s<numStates; s++)
		{
			for(int i=0;i<numActions; i++)
			{
				System.out.println("For state "+s+" with data "+s/51+" and energy "+ s%51+" and action "+i+ " the value is "+valuefunction[i][s]);
			}
		}
	}

} 