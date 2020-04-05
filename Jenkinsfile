pipeline {
    agent { docker{
        label 'Docker'
        image 'python:3.7.2' 
    } }
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
                sh 'python3 -m unittest -v test/sim_test.py'
            }
        }
    }
}
