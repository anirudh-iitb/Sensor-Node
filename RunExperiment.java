import org.rlcommunity.rlglue.codec.LocalGlue;
import org.rlcommunity.rlglue.codec.RLGlue;
import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.util.EnvironmentLoader;
import org.rlcommunity.rlglue.codec.util.AgentLoader;
import org.rlcommunity.rlglue.codec.EnvironmentInterface;

public class RunExperiment{
	
	public static void main(String[] args){
		//Create the Agent
		AgentInterface theAgent=new SensorAgent();
		
		//Create the Environment
		EnvironmentInterface theEnvironment=new SensorEnvironment();
		
		LocalGlue localGlueImplementation=new LocalGlue(theEnvironment,theAgent);
		RLGlue.setGlue(localGlueImplementation);
		
		
		//Run the main method of the Sample Experiment, using the arguments were were passed
		//This will run the experiment in the main thread.  The Agent and Environment will run
		//locally, without sockets.
		SensorExperiment.main(args);
		System.out.println("RunExperiment Complete");
		
	}

}