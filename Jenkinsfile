pipeline {
    agent { docker { image 'python:3.7.3' } }
    stages {
        stage('build') {
            steps {
                echo python --version
            }
        }
    }
}