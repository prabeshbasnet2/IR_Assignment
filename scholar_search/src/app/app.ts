import { Component } from '@angular/core';
import { SearchComponent } from './components/search/search.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [SearchComponent],   // 👈 import child standalone component here
  template: `
    <app-search></app-search>
  `,
  styleUrls: ['./app.css']
})
export class AppComponent {}
