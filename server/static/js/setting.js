import Vue from 'vue';
import HTTP from './http';

const vm = new Vue({
    el: '#mail-app',
    delimiters: ['[[', ']]'],
    data () {
        return {
            id: null,
            active_learning_strategy: null,
            sampling_threshold: 0.5,
            proactive_learning_strategy: null,
            proficiency_threshold: 0.5,
            steps_before_retraining: null,
            allocate_new_batch: null,
            samples_per_session: null,
            active_learning_options: [{id: 1, name: 'Random'}, {id: 2, name: 'Hardest Samples First'}, {id: 3, name: 'Maintain Class Balance'}],
            proactive_learning_options: [{id: 1, name: 'Random'}, {id: 2, name: 'Less Expensive Annotators First'}, {id: 3, name: 'Best Annotators First'}],
            allocate_new_batch_options: [{id: 1, name: 'When all annotators finish annotations'}, {id: 2, name: 'When any one annotator finishes annotations'}],
        }
    },
    methods: {
        save() {
            const payload = {
                id: this.id,
                active_learning_strategy: this.active_learning_strategy,
                sampling_threshold: this.sampling_threshold,
                proactive_learning_strategy: this.proactive_learning_strategy,
                proficiency_threshold: this.proficiency_threshold,
                steps_before_retraining: this.steps_before_retraining,
                samples_per_session: this.samples_per_session,
                allocate_new_batch: this.allocate_new_batch,
            };
            HTTP.patch('/', payload).then((response) => {});
        },
        reset() {
            HTTP.get('').then((response) => {
                this.active_learning_strategy = response.data.active_learning_strategy;
                this.sampling_threshold = response.data.sampling_threshold;
                this.proactive_learning_strategy = response.data.proactive_learning_strategy;
                this.proficiency_threshold = response.data.proficiency_threshold;
                this.steps_before_retraining = response.data.steps_before_retraining;
                this.samples_per_session = response.data.samples_per_session;
                this.allocate_new_batch = response.data.allocate_new_batch;
            });
        }
    },
    created() {
        console.log('Called from created');
        HTTP.get('').then((response) => {
            this.id = response.data.id;
            this.steps_before_retraining = response.data.steps_before_retraining;
            this.samples_per_session = response.data.samples_per_session;
            this.active_learning_strategy = response.data.active_learning_strategy;
            this.sampling_threshold = response.data.sampling_threshold;
            this.proficiency_threshold = response.data.proficiency_threshold;
            this.proactive_learning_strategy = response.data.proactive_learning_strategy;
            this.allocate_new_batch = response.data.allocate_new_batch;
        });
    },
});