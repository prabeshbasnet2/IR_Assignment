import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { SearchService } from './search.service';

describe('SearchService', () => {
  let service: SearchService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [SearchService]
    });
    service = TestBed.inject(SearchService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should call /search API', () => {
    const mockResponse = { results: [], page: 1, size: 10, totalData: 0 };

    service.search('test').subscribe((res) => {
      expect(res).toEqual(mockResponse);
    });

    const req = httpMock.expectOne((r) => r.url.includes('/search'));
    expect(req.request.method).toBe('GET');
    req.flush(mockResponse);
  });

  it('should call /classify API', () => {
    const mockResponse = { label: 'business', confidence: 0.95 };

    service.classify('Stock market news').subscribe((res) => {
      expect(res).toEqual(mockResponse);
    });

    const req = httpMock.expectOne((r) => r.url.includes('/classify'));
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ text: 'Stock market news' });
    req.flush(mockResponse);
  });
});
