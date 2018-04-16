from smurff import PredictSession 
import sys

session = PredictSession.fromRootFile(sys.argv[1])
print(session)