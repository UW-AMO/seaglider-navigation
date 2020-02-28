pipeline {
    agent { docker { image 'python:3.7.3' } }
    stages {
        stage('build') {
            steps {
                echo python --version
                pip install requirements.txt
                pip install -e .
            }
        }
	    stage('test') {
            steps {
                python -m unittest -v test/test.py
            }
        }
        stage('coverage') {
            steps {
                echo 'Code coverage not yet implemented.'
            }
        }
    }
}