pipeline {
    agent any 
    stages {
        stage('build') {
            steps {
                sh 'python --version'
                sh 'pip install -e .'
            }
        }
        stage('test') {
            steps {
                sh 'python -m unittest -v test/test.py'
            }
        }
    }
}