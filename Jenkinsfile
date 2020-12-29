pipeline {
    agent { docker{
        label 'Docker'
        image 'python:3.7.7'
    } }
    stages {
        stage('build') {
            steps {
                sh 'python3 --version'
                sh 'pip3 install -e .'
                sh 'pip install -r requirements-dev.txt'
                sh 'pre-commit install'
            }
        }
        stage('test') {
            steps {
                sh 'pre-commit run --all-files'
                sh 'coverage run -m pytest adcp'
                sh 'coverage report'
            }
        }
    }
}
