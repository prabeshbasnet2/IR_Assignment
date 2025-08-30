import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SearchService {
  private apiUrl = 'http://127.0.0.1:8000'; // backend FastAPI URL

  constructor(private http: HttpClient) {}

  // ---------- SEARCH ----------
  search(query: string, page: number = 1, size: number = 10): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/search`, {
      params: { query, page, size }
    });
  }

  // ---------- CLASSIFY ----------
  classify(text: string): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/classify`, { text });
  }
}
