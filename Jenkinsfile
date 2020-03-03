pipeline {
    agent any 
    stages {
        stage('build') {
            steps {
                sh 'python3 --version'
                sh 'pip3 install -e .'
            }
        }
        stage('test') {
            steps {
                sh 'python3 -m unittest -v test/test.py'
            }
        }
    }
}