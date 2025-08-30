import { Component } from '@angular/core';
import { CommonModule, NgIf } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SearchService } from '../../services/search.service';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [CommonModule, FormsModule, NgIf],
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent {
  // --- Tab control ---
  activeTab: 'search' | 'classify' = 'search';

  // --- Search ---
  query = '';
  results: any[] = [];
  loading = false;
  error = '';
  currentPage = 1;
  pageSize = 10;
  hasMoreResults = false;

  // --- Classification ---
  classifyText = '';
  classifyResult: any = null;
  classifyLoading = false;
  classifyError = '';

  constructor(private searchService: SearchService) {}

  // SEARCH
  onSearch() {
    if (!this.query.trim()) return;
    this.loading = true;
    this.error = '';
    this.results = [];

    this.searchService.search(this.query, this.currentPage, this.pageSize).subscribe({
      next: (data) => {
        this.results = data.results.map((item: any) => ({
          ...item,
          showFullAbstract: false, // collapsed by default
        }));
        this.hasMoreResults = (this.currentPage * this.pageSize) < data.totalData;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Error fetching results';
        console.error(err);
        this.loading = false;
      }
    });
  }

  nextPage() {
    if (!this.hasMoreResults) return;
    this.currentPage++;
    this.onSearch();
  }

  prevPage() {
    if (this.currentPage > 1) {
      this.currentPage--;
      this.onSearch();
    }
    
  }

  // CLASSIFY
  onClassify() {
    if (!this.classifyText.trim()) return;
    this.classifyLoading = true;
    this.classifyError = '';
    this.classifyResult = null;

    this.searchService.classify(this.classifyText).subscribe({
      next: (res) => {
        this.classifyResult = res;
        this.classifyLoading = false;
      },
      error: (err) => {
        this.classifyError = 'Error classifying text';
        console.error(err);
        this.classifyLoading = false;
      }
    });
  }
}
