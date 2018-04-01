login:
	ssh -i vesselClassifier.pem ubuntu@ec2-34-216-14-129.us-west-2.compute.amazonaws.com

fetch:
	scp -i vesselClassifier.pem ubuntu@ec2-34-216-14-129.us-west-2.compute.amazonaws.com:~/code/vesselSegmentation/images/LossAndCorrect.png images/
